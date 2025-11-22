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
    # 问题1: 每次循环都计算最小值（可优化为只计算一次）
    min_abundance_list = []
    for idx in range(len(df_raw)):
        min_abundance_list.append(df_raw['Abundance'].min())
    min_abundance = min_abundance_list[0]
    
    # --- 1. MPERI Calculation ---
    st.markdown("### 1. MPERI (Microplastics Environmental Risk Index) Calculation")
    
    with st.expander("View MPERI Formula Details"):
        st.markdown(r"""
**Formula:**
$MPERI = PLI \times T_m$

1. **PLI (Pollution Load Index):**
   $\sqrt{Abundance / C_{ref}}$
""")
        st.markdown(f" * **$C_{{ref}}$** (Reference Value) is set to the **minimum abundance** in the dataset: **{min_abundance}**")
        st.markdown(r"""
2. **Tm (Comprehensive Characteristic Coefficient):**
   $[T_{polymer} \times (0.2 \cdot T_{color} + 0.8 \cdot T_{shape})] \times P_{size}$

**Weights:**
* $T_{polymer}$: PP(1), PET(4), PE(11), PS(30)
* $T_{color}$: Yellow/Red(1.0), Blue/Brown/Black(0.66), White(0.33)
* $T_{shape}$: Fiber(1.0), Film(0.88), Fragment(0.83), Pellet(0.65)
* $P_{size}$: Proportion of 0-1000μm
""")
    
    try:
        # --- 1.1 Calculate PLI ---
        # 问题2: 使用循环逐行计算（可优化为向量化操作）
        pli_list = []
        for i in range(len(df_raw)):
            abundance_val = df_raw.iloc[i]['Abundance']
            pli_val = np.sqrt(abundance_val / min_abundance)
            pli_list.append(pli_val)
        df_raw['PLI'] = pli_list
        
        # --- 1.2 Calculate Tm Components ---
        # 问题3: 重复计算分母（可优化为先计算分母再复用）
        t_polymer_list = []
        for i in range(len(df_raw)):
            numerator = (df_raw.iloc[i]['PP']*1 + df_raw.iloc[i]['PET']*4 + 
                        df_raw.iloc[i]['PE']*11 + df_raw.iloc[i]['PS']*30)
            denominator = (df_raw.iloc[i]['PP'] + df_raw.iloc[i]['PET'] + 
                          df_raw.iloc[i]['PE'] + df_raw.iloc[i]['PS'])
            t_polymer_list.append(numerator / denominator)
        df_raw['T_polymer'] = t_polymer_list
        
        # 问题4: 多次访问DataFrame（可优化为一次性提取）
        t_color_list = []
        for i in range(len(df_raw)):
            yellow = df_raw.iloc[i]['Yellow']
            red = df_raw.iloc[i]['Red']
            blue = df_raw.iloc[i]['Blue']
            brown = df_raw.iloc[i]['Brown']
            black = df_raw.iloc[i]['Black']
            white = df_raw.iloc[i]['White']
            numerator = (yellow + red)*1.0 + (blue + brown + black)*0.66 + white*0.33
            denominator = yellow + red + blue + brown + black + white
            t_color_list.append(numerator / denominator)
        df_raw['T_color'] = t_color_list
        
        # 问题5: 同样的冗余模式
        t_shape_list = []
        for i in range(len(df_raw)):
            fibre = df_raw.iloc[i]['Fibre']
            film = df_raw.iloc[i]['Film']
            fragment = df_raw.iloc[i]['Fragment']
            pellet = df_raw.iloc[i]['Pellet']
            numerator = fibre*1.0 + film*0.88 + fragment*0.83 + pellet*0.65
            denominator = fibre + film + fragment + pellet
            t_shape_list.append(numerator / denominator)
        df_raw['T_shape'] = t_shape_list
        
        # 问题6: Size计算也用循环
        p_size_list = []
        for i in range(len(df_raw)):
            small = df_raw.iloc[i]['＜100um']
            medium = df_raw.iloc[i]['100um-500um']
            large = df_raw.iloc[i]['500um-1000um']
            xlarge = df_raw.iloc[i]['1000um-5000um']
            numerator = small + medium + large
            denominator = small + medium + large + xlarge
            p_size_list.append(numerator / denominator)
        df_raw['P_size'] = p_size_list
        
        # --- 1.3 Calculate Tm & MPERI ---
        tm_list = []
        for i in range(len(df_raw)):
            t_poly = df_raw.iloc[i]['T_polymer']
            t_col = df_raw.iloc[i]['T_color']
            t_shp = df_raw.iloc[i]['T_shape']
            p_sz = df_raw.iloc[i]['P_size']
            tm_val = (t_poly * (0.2 * t_col + 0.8 * t_shp)) * p_sz
            tm_list.append(tm_val)
        df_raw['Tm'] = tm_list
        
        mperi_list = []
        for i in range(len(df_raw)):
            pli = df_raw.iloc[i]['PLI']
            tm = df_raw.iloc[i]['Tm']
            mperi_list.append(pli * tm)
        df_raw['MPERI'] = mperi_list
        
        # --- Display Results ---
        res_df = df_raw[['Sites', 'Abundance', 'PLI', 'MPERI']]
        st.write("MPERI Calculation Results:")
        st.dataframe(res_df.head())
        
        # Visualization
        fig_risk = px.bar(
            res_df.groupby('Sites')['MPERI'].mean().reset_index(),
            x='Sites', y='MPERI', color='MPERI',
            title='Average MPERI by Site',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
    except Exception as e:
        st.error(f"MPERI Calculation Failed: {e}")
        st.stop()
    
    # --- 2. Machine Learning ---
    st.markdown("---")
    st.markdown("### 2. Machine Learning (Risk Prediction)")
    st.info("Using features (Abundance + Table 2 components) to predict MPERI.")
    
    # --- Construct Feature Set ---
    try:
        features_df = pd.DataFrame()
        
        # 问题7: 特征构建也用循环（可向量化）
        abundance_feat = []
        for i in range(len(df_raw)):
            abundance_feat.append(df_raw.iloc[i]['Abundance'])
        features_df['Abundance'] = abundance_feat
        
        size_0_1 = []
        for i in range(len(df_raw)):
            val = (df_raw.iloc[i]['＜100um'] + df_raw.iloc[i]['100um-500um'] + 
                   df_raw.iloc[i]['500um-1000um'])
            size_0_1.append(val)
        features_df['Size_0-1mm'] = size_0_1
        
        size_1_5 = []
        for i in range(len(df_raw)):
            size_1_5.append(df_raw.iloc[i]['1000um-5000um'])
        features_df['Size_1-5mm'] = size_1_5
        
        color_white = []
        for i in range(len(df_raw)):
            color_white.append(df_raw.iloc[i]['White'])
        features_df['Color_White'] = color_white
        
        color_black = []
        for i in range(len(df_raw)):
            color_black.append(df_raw.iloc[i]['Black'])
        features_df['Color_Black'] = color_black
        
        color_colored = []
        for i in range(len(df_raw)):
            val = (df_raw.iloc[i]['Brown'] + df_raw.iloc[i]['Blue'] + 
                   df_raw.iloc[i]['Red'] + df_raw.iloc[i]['Yellow'])
            color_colored.append(val)
        features_df['Color_Colored'] = color_colored
        
        poly_pet = []
        for i in range(len(df_raw)):
            poly_pet.append(df_raw.iloc[i]['PET'])
        features_df['Poly_PET'] = poly_pet
        
        poly_ps = []
        for i in range(len(df_raw)):
            poly_ps.append(df_raw.iloc[i]['PS'])
        features_df['Poly_PS'] = poly_ps
        
        poly_pp = []
        for i in range(len(df_raw)):
            poly_pp.append(df_raw.iloc[i]['PP'])
        features_df['Poly_PP'] = poly_pp
        
        poly_pe = []
        for i in range(len(df_raw)):
            poly_pe.append(df_raw.iloc[i]['PE'])
        features_df['Poly_PE'] = poly_pe
        
        shape_fibre = []
        for i in range(len(df_raw)):
            shape_fibre.append(df_raw.iloc[i]['Fibre'])
        features_df['Shape_Fibre'] = shape_fibre
        
        shape_pellet = []
        for i in range(len(df_raw)):
            shape_pellet.append(df_raw.iloc[i]['Pellet'])
        features_df['Shape_Pellet'] = shape_pellet
        
        shape_film = []
        for i in range(len(df_raw)):
            shape_film.append(df_raw.iloc[i]['Film'])
        features_df['Shape_Film'] = shape_film
        
        shape_fragment = []
        for i in range(len(df_raw)):
            shape_fragment.append(df_raw.iloc[i]['Fragment'])
        features_df['Shape_Fragment'] = shape_fragment
        
        X = features_df
        y = df_raw['MPERI']
        
        # --- ML Sidebar Configuration ---
        st.sidebar.header("ML Configuration")
        model_type = st.sidebar.selectbox("Select Algorithm", 
                                         ["LightGBM", "GradientBoostingRegressor"])
        test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2)
        
        if model_type == "LightGBM":
            n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100)
            lr = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.1)
            num_leaves = st.sidebar.slider("num_leaves", 10, 100, 31)
        else:
            n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100)
            lr = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.1)
            max_depth = st.sidebar.slider("max_depth", 3, 10, 3)
        
        # --- Run Training ---
        if st.button("Start Training & Analysis"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)
            
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
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            c1, c2 = st.columns(2)
            c1.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
            c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
            
            st.subheader("Feature Importance Ranking")
            imp = model.feature_importances_
            feat_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': imp
            }).sort_values(by='Importance', ascending=False)
            
            st.plotly_chart(px.bar(feat_df, x='Importance', y='Feature', 
                                  orientation='h', title="Feature Importance"),
                          use_container_width=True)
            
    except Exception as e:
        st.error(f"Machine Learning Module Error: {e}")
