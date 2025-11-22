import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

st.set_page_config(page_title="Risk Assessment", layout="wide")
st.title("⚠️ Module 3: Risk")

if 'raw_data' not in st.session_state:
    st.error("Please upload data on the Home page first.")
else:
    df_raw = st.session_state['raw_data'].copy()

    # --- Calculate Reference Value (Minimum Abundance) ---
    min_abundance = df_raw['Abundance'].min()

    # --- 1. MPERI Calculation ---
    st.markdown("### 1. MPERI (Microplastics Environmental Risk Index) Calculation")
    
    with st.expander("View MPERI Formula Details"):
        # Part 1: Formula (Raw string 'r')
        st.markdown(r"""
        **Formula:** $MPERI = PLI \times T_m$
        
        1. **PLI (Pollution Load Index):** $\sqrt{Abundance / C_{ref}}$
        """)
        
        # Part 2: Dynamic Value (F-string)
        st.markdown(f"   * **$C_{{ref}}$** (Reference Value) is set to the **minimum abundance** in the dataset: **{min_abundance}**")
        
        # Part 3: Tm Formula (Raw string 'r')
        st.markdown(r"""
        2. **Tm (Comprehensive Characteristic Coefficient):** $[T_{polymer} \times (0.2 \cdot T_{color} + 0.8 \cdot T_{shape})] \times P_{size}$
        
        **Weights:**
        * $T_{polymer}$: PP(1), PET(4), PE(11), PS(30)
        * $T_{color}$: Yellow/Red(1.0), Blue/Brown/Black(0.66), White(0.33)
        * $T_{shape}$: Fiber(1.0), Film(0.88), Fragment(0.83), Pellet(0.65)
        * $P_{size}$: Proportion of 0-1000μm
        """)

    try:
        # --- 1.1 Calculate PLI ---
        # PLI = sqrt(Abundance / Minimum_Abundance)
        df_raw['PLI'] = np.sqrt(df_raw['Abundance'] / min_abundance)

        # --- 1.2 Calculate Tm Components (修改为加权和 / 总和) ---
        
        # Polymer Score
        # 修改：(各聚合物加权和) / (各聚合物总和)
        df_raw['T_polymer'] = (df_raw['PP']*1 + df_raw['PET']*4 + df_raw['PE']*11 + df_raw['PS']*30) / (df_raw['PP'] + df_raw['PET'] + df_raw['PE'] + df_raw['PS'])
        
        # Color Score
        # 修改：(各颜色加权和) / (各颜色总和)
        df_raw['T_color'] = ((df_raw['Yellow']+df_raw['Red'])*1.0 + (df_raw['Blue']+df_raw['Brown']+df_raw['Black'])*0.66 + df_raw['White']*0.33) / (df_raw['Yellow'] + df_raw['Red'] + df_raw['Blue'] + df_raw['Brown'] + df_raw['Black'] + df_raw['White'])
        
        # Shape Score
        # 修改：(各形状加权和) / (各形状总和)
        df_raw['T_shape'] = (df_raw['Fibre']*1.0 + df_raw['Film']*0.88 + df_raw['Fragment']*0.83 + df_raw['Pellet']*0.65) / (df_raw['Fibre'] + df_raw['Film'] + df_raw['Fragment'] + df_raw['Pellet'])
        
        # Size Proportion (0-1mm)
        # 修改：(0-1mm尺寸和) / (总尺寸和)
        df_raw['P_size'] = (df_raw['＜100um'] + df_raw['100um-500um'] + df_raw['500um-1000um']) / (df_raw['＜100um'] + df_raw['100um-500um'] + df_raw['500um-1000um'] + df_raw['1000um-5000um'])

        # --- 1.3 Calculate Tm & MPERI ---
        df_raw['Tm'] = (df_raw['T_polymer'] * (0.2 * df_raw['T_color'] + 0.8 * df_raw['T_shape'])) * df_raw['P_size']
        df_raw['MPERI'] = df_raw['PLI'] * df_raw['Tm']

        # --- Display Results (Removed Risk_Level) ---
        res_df = df_raw[['Sites', 'Abundance', 'PLI', 'MPERI']]
        st.write("MPERI Calculation Results:")
        st.dataframe(res_df.head())
        
        # Visualization
        fig_risk = px.bar(
            res_df.groupby('Sites')['MPERI'].mean().reset_index(), 
            x='Sites', y='MPERI', color='MPERI', 
            title='Average MPERI by Site', color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    except Exception as e:
        st.error(f"MPERI Calculation Failed: {e}")
        st.stop()

    # --- 2. Machine Learning ---
    st.markdown("---")
    st.markdown("### 2. Machine Learning (Risk Prediction)")
    st.info("Using features (Abundance + Table 2 components) to predict MPERI.")

    # --- Construct Feature Set (Data Preparation) ---
    try:
        features_df = pd.DataFrame()
        
        # Add Abundance
        features_df['Abundance'] = df_raw['Abundance']
        
        # Size Features
        features_df['Size_0-1mm'] = df_raw['＜100um'] + df_raw['100um-500um'] + df_raw['500um-1000um']
        features_df['Size_1-5mm'] = df_raw['1000um-5000um']
        # Color Features
        features_df['Color_White'] = df_raw['White']
        features_df['Color_Black'] = df_raw['Black']
        features_df['Color_Colored'] = df_raw['Brown'] + df_raw['Blue'] + df_raw['Red'] + df_raw['Yellow']
        # Polymer Features
        features_df['Poly_PET'] = df_raw['PET']
        features_df['Poly_PS'] = df_raw['PS']
        features_df['Poly_PP'] = df_raw['PP']
        features_df['Poly_PE'] = df_raw['PE']
        # Shape Features
        features_df['Shape_Fibre'] = df_raw['Fibre']
        features_df['Shape_Pellet'] = df_raw['Pellet']
        features_df['Shape_Film'] = df_raw['Film']
        features_df['Shape_Fragment'] = df_raw['Fragment']

        # Define X (Features) and y (Target)
        X = features_df
        y = df_raw['MPERI']

        # --- ML Sidebar Configuration ---
        st.sidebar.header("ML Configuration")
        model_type = st.sidebar.selectbox("Select Algorithm", ["LightGBM", "GradientBoostingRegressor"])
        test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2)
        
        if model_type == "LightGBM":
            n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100)
            lr = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.1)
            num_leaves = st.sidebar.slider("num_leaves", 10, 100, 31)
        else: # GradientBoostingRegressor
            n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100)
            lr = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.1)
            max_depth = st.sidebar.slider("max_depth", 3, 10, 3)

        # --- Run Training ---
        if st.button("Start Training & Analysis"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            if model_type == "LightGBM":
                model = lgb.LGBMRegressor(
                    n_estimators=n_estimators, 
                    learning_rate=lr, 
                    num_leaves=num_leaves, 
                    random_state=42,
                    verbose=-1
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=n_estimators, 
                    learning_rate=lr, 
                    max_depth=max_depth, 
                    random_state=42
                )
            
            # Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metrics
            c1, c2 = st.columns(2)
            c1.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
            c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
            
            # Feature Importance
            st.subheader("Feature Importance Ranking")
            imp = model.feature_importances_
            feat_df = pd.DataFrame({
                'Feature': X.columns, 
                'Importance': imp
            }).sort_values(by='Importance', ascending=False)
            
            st.plotly_chart(px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Feature Importance"), use_container_width=True)

    except Exception as e:
        st.error(f"Machine Learning Module Error: {e}")
