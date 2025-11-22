import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Distribution Analysis", layout="wide")
st.title("📊 Module 1: Distribution")

if 'raw_data' not in st.session_state:
    st.error("Please upload data on the Home page first.")
else:
    df = st.session_state['raw_data'].copy()
    
    # --- 1. Average Abundance ---
    st.markdown("### 1. Average Abundance")
    
    # Calculate mean for numeric columns only
    abundance_mean = df.groupby('Sites')['Abundance'].mean().reset_index()
    
    col_a, col_b = st.columns([1, 3])
    with col_a:
        chart_type_ab = st.radio("Chart Type", ['Bar Chart', 'Box Plot'], key="ab_chart")
    with col_b:
        if chart_type_ab == 'Bar Chart':
            fig = px.bar(abundance_mean, x='Sites', y='Abundance', title='Average Abundance by Site')
        else:
            fig = px.box(df, x='Sites', y='Abundance', title='Abundance Distribution by Site')
        st.plotly_chart(fig, use_container_width=True)

    # --- 2. Property Proportions ---
    st.markdown("---")
    st.markdown("### 2. Property Proportions")
    
    # Map English UI names to Data Columns
    # Ensure these column names match your Excel file exactly
    cols_dict = {
        'Size': ['＜100um', '100um-500um', '500um-1000um', '1000um-5000um'],
        'Polymer': ['PS', 'PET', 'PP', 'PE', 'Others'],
        'Shape': ['Fibre', 'Pellet', 'Film', 'Fragment'],
        'Color': ['Black', 'Brown', 'White', 'Blue', 'Red', 'Yellow']
    }

    c1, c2 = st.columns(2)
    with c1:
        for name in ['Size', 'Shape']:
            # Calculate mean proportion
            avg = df[cols_dict[name]].mean()
            st.plotly_chart(px.pie(values=avg, names=avg.index, title=f"{name} Proportion"), use_container_width=True)
    with c2:
        for name in ['Polymer', 'Color']:
            avg = df[cols_dict[name]].mean()
            st.plotly_chart(px.pie(values=avg, names=avg.index, title=f"{name} Proportion"), use_container_width=True)

    # --- 3. Diversity Indices & MDII ---
    st.markdown("---")
    st.markdown("### 3. Diversity Indices & MDII")

    try:
        # Pre-calculate Simpson Index for each sample (row)
        diversity_df = df[['Sites']].copy()
        d_col_map = {} 

        for category, cols in cols_dict.items():
            # Normalize to 0-1 (assuming input is percentage 0-100)
            props = df[cols].fillna(0) / 100.0
            # Sum(pi^2)
            sum_sq = (props ** 2).sum(axis=1)
            # D = 1 - Sum(pi^2)
            col_name = f'D_{category}' 
            diversity_df[col_name] = 1 - sum_sq
            d_col_map[category] = col_name

        # --- 3.1 Individual Simpson Indices (Tabs) ---
        st.subheader("3.1 Simpson Diversity Index by Property")
        st.markdown("Formula: $D = 1 - \sum p_i^2$")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Size", "Polymer", "Shape", "Color"])
        
        def plot_diversity(tab, category_name, data_col):
            with tab:
                c_opt, c_plot = st.columns([1, 4])
                with c_opt:
                    ctype = st.radio(f"Chart Type ({category_name})", ['Bar Chart (Mean)', 'Box Plot (Dist.)'], key=f"d_{data_col}")
                
                with c_plot:
                    if 'Bar' in ctype:
                        d_mean = diversity_df.groupby('Sites')[data_col].mean().reset_index()
                        fig = px.bar(
                            d_mean, x='Sites', y=data_col, 
                            title=f'{category_name} Simpson Diversity (Mean)',
                            color=data_col, color_continuous_scale='Viridis'
                        )
                    else:
                        fig = px.box(
                            diversity_df, x='Sites', y=data_col, 
                            title=f'{category_name} Simpson Diversity (Distribution)',
                            color='Sites'
                        )
                    st.plotly_chart(fig, use_container_width=True)

        plot_diversity(tab1, "Size", d_col_map['Size'])
        plot_diversity(tab2, "Polymer", d_col_map['Polymer'])
        plot_diversity(tab3, "Shape", d_col_map['Shape'])
        plot_diversity(tab4, "Color", d_col_map['Color'])

        # --- 3.2 MDII Calculation ---
        st.markdown("---")
        st.subheader("3.2 Microplastic Diversity Integrated Index (MDII)")
        st.info("Formula: $MDII = (D_{size} \times D_{polymer} \times D_{shape} \times D_{color})^{1/4}$")

        # Calculation
        product_d = (
            diversity_df[d_col_map['Size']] * diversity_df[d_col_map['Polymer']] * diversity_df[d_col_map['Shape']] * diversity_df[d_col_map['Color']]
        )
        diversity_df['MDII'] = product_d ** 0.25
        
        # Summary
        mdii_summary = diversity_df.groupby('Sites')['MDII'].mean().reset_index()
        
        # Display Table
        st.write("MDII Results per Site:")
        # Apply formatting only to the MDII column
        st.dataframe(mdii_summary.style.format({"MDII": "{:.4f}"}))
        
        # Display Chart
        fig_mdii = px.bar(
            mdii_summary, x='Sites', y='MDII', 
            title='MDII by Site', 
            color='MDII', color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_mdii, use_container_width=True)

    except Exception as e:
        st.error(f"Calculation Error: {e}")
        st.warning("Please check if your data columns match the requirements.")