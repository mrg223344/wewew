import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

# --- lightgbm optional import: fallback silently to sklearn if not installed ---
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except Exception:
    _LGBM_AVAILABLE = False

    from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor

    class _LGBMRegressorFallback:
        """LightGBM fallback wrapper using sklearn gradient boosting."""
        def __init__(self, n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=None, verbose=-1):
            try:
                self.model = HistGradientBoostingRegressor(
                    max_iter=int(n_estimators),
                    learning_rate=float(learning_rate),
                    random_state=random_state
                )
            except Exception:
                self.model = GradientBoostingRegressor(
                    n_estimators=int(n_estimators),
                    learning_rate=float(learning_rate),
                    random_state=random_state,
                    max_depth=3
                )
            self._n_features = None

        def fit(self, X, y):
            self.model.fit(X, y)
            try:
                self._n_features = X.shape[1]
            except Exception:
                self._n_features = None
            return self

        def predict(self, X):
            return self.model.predict(X)

        @property
        def feature_importances_(self):
            if hasattr(self.model, "feature_importances_"):
                return self.model.feature_importances_
            else:
                import numpy as _np
                if self._n_features is not None:
                    return _np.zeros(self._n_features)
                return _np.array([])

    class _lgb_module_like:
        LGBMRegressor = _LGBMRegressorFallback
    lgb = _lgb_module_like()


# --- Streamlit App ---
st.set_page_config(page_title="Risk Assessment", layout="wide")
st.title("⚠️ Module 3: Risk")

if 'raw_data' not in st.session_state:
    st.error("Please upload data on the Home page first.")
else:
    df_raw = st.session_state['raw_data'].copy()

    # --- Calculate Reference Value ---
    min_abundance = df_raw['Abundance'].min()

    # --- 1. MPERI Calculation ---
    st.markdown("### 1. MPERI (Microplastics Environmental Risk Index) Calculation")
    
    with st.expander("View MPERI Formula Details"):
        st.markdown(r"""
        **Formula:** $MPERI = PLI \times T_m$
        
        1. **PLI (Pollution Load Index):** $\sqrt{Abundance / C_{ref}}$
        """)
        st.markdown(f"   * **$C_{{ref}}$** (Reference Value) is set to the **minimum abundance** in the dataset: **{min_abundance}**")
        st.markdown(r"""
        2. **Tm (Comprehensive Characteristic Coefficient):** $[T_{polymer} \times (0.2 \cdot T_{color} + 0.8 \cdot T_{shape})] \times P_{size}$
        
        **Weights:**
        * $T_{polymer}$: PP(1), PET(4), PE(11), PS(30)
        * $T_{color}$: Yellow/Red(1.0), Blue/Brown/Black(0.66), White(0.33)
        * $T_{shape}$: Fiber(1.0), Film(0.88), Fragment(0.83), Pellet(0.65)
        * $P_{size}$: Proportion of 0-1000μm
        """)

    try:
        # --- PLI ---
        df_raw['PLI'] = np.sqrt(df_raw['Abundance'] / min_abundance)

        # --- Weighted Scores ---
        df_raw['T_polymer'] = (df_raw['PP']*1 + df_raw['PET']*4 + df_raw['PE']*11 + df_raw['PS']*30) / (df_raw['PP'] + df_raw['PET'] + df_raw['PE'] + df_raw['PS'])
        df_raw['T_color'] = ((df_raw['Yellow']+df_raw['Red'])*1.0 + (df_raw['Blue']+df_raw['Brown']+df_raw['Black'])*0.66 + df_raw['White']*0.33) / (df_raw['Yellow'] + df_raw['Red'] + df_raw['Blue'] + df_raw['Brown'] + df_raw['Black'] + df_raw['White'])
        df_raw['T_shape'] = (df_raw['Fibre']*1.0 + df_raw['Film']*0.88 + df_raw['Fragment']*0.83 + df_raw['Pellet']*0.65) / (df_raw['Fibre'] + df_raw['Film'] + df_raw['Fragment'] + df_raw['Pellet'])
        df_raw['P_size'] = (df_raw['＜100um'] + df_raw['100um-500um'] + df_raw['500um-1000um']) / (df_raw['＜100um'] + df_raw['100um-500um'] + df_raw['500um-1000um'] + df_raw['1000um-5000um'])

        df_raw['Tm'] = (df_raw['T_polymer'] * (0.2 * df_raw['T_color'] + 0.8 * df_raw['T_shape'])) * df_raw['P_size']
        df_raw['MPERI'] = df_raw['PLI'] * df_raw['Tm']

        res_df = df_raw[['Sites', 'Abundance', 'PLI', 'MPERI']]
        st.write("MPERI Calculation Results:")
        st.dataframe(res_df.head())

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

    try:
        features_df = pd.DataFrame()
        features_df['Abundance'] = df_raw['Abundance']
        features_df['Size_0-1mm'] = df_raw['＜100um'] + df_raw['100um-500um'] + df_raw['500um-1000um']
        features_df['Size_1-5mm'] = df_raw['1000um-5000um']
        features_df['Color_White'] = df_raw['White']
        features_df['Color_Black'] = df_raw['Black']
        features_df['Color_Colored'] = df_raw['Brown'] + df_raw['Blue'] + df_raw['Red'] + df_raw['Yellow']
        features_df['Poly_PET'] = df_raw['PET']
        features_df['Poly_PS'] = df_raw['PS']
        features_df['Poly_PP'] = df_raw['PP']
        features_df['Poly_PE'] = df_raw['PE']
        features_df['Shape_Fibre'] = df_raw['Fibre']
        features_df['Shape_Pellet'] = df_raw['Pellet']
        features_df['Shape_Film'] = df_raw['Film']
        features_df['Shape_Fragment'] = df_raw['Fragment']

        X = features_df
        y = df_raw['MPERI']

        st.sidebar.header("ML Configuration")
        model_type = st.sidebar.selectbox("Select Algorithm", ["LightGBM", "GradientBoostingRegressor"])
        test_size = st.sidebar.slide_
