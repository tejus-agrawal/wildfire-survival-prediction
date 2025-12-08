import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

# Page Config
st.set_page_config(page_title="Wildfire Survival Prediction", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Try loading from local path
    path = "app_data.csv"
    if not os.path.exists(path):
        # Fallback for dev environment structure
        path = "../app/app_data.csv"
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else:
        return None

import os
df = load_data()

# --- SIDEBAR ---
st.sidebar.title("🔥 Palisades Wildfire Model")
st.sidebar.markdown("Predicting home survival probability using Satellite Imagery & ML.")

if df is not None:
    # Filters
    st.sidebar.header("Filters")
    risk_filter = st.sidebar.multiselect(
        "Risk Factor", 
        options=df['risk_factor'].unique(),
        default=df['risk_factor'].unique()
    )
    
    filtered_df = df[df['risk_factor'].isin(risk_filter)]
    
    # Show stats
    st.sidebar.markdown("---")
    st.sidebar.metric("Homes Analyzed", len(filtered_df))
    avg_prob = filtered_df['xgb_prob'].mean()
    st.sidebar.metric("Avg. Burn Probability", f"{avg_prob:.1%}")

# --- MAIN CONTENT ---
st.title("🏡 Wildfire Risk Map")

if df is None:
    st.error("Data not found. Please run 'notebooks/app_preparation.ipynb' to generate 'app_data.csv'.")
else:
    # Map
    # Color mapping
    # Red for high risk, Green for low risk
    # We'll visualize 'xgb_prob' (0 to 1)
    
    filtered_df['color_r'] = (filtered_df['xgb_prob'] * 255).astype(int)
    filtered_df['color_g'] = ((1 - filtered_df['xgb_prob']) * 255).astype(int)
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        filtered_df,
        get_position=["lon", "lat"],
        get_color=["color_r", "color_g", 0, 160],
        get_radius=10,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=filtered_df['lat'].mean(),
        longitude=filtered_df['lon'].mean(),
        zoom=12,
        pitch=0,
    )

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "ID: {id}\nRisk: {risk_factor}\nBurn Prob: {xgb_prob}"},
    )

    st.pydeck_chart(r)
    
    # Data Table
    st.subheader("High Risk Homes")
    high_risk = filtered_df[filtered_df['xgb_prob'] > 0.7].sort_values('xgb_prob', ascending=False).head(10)
    st.dataframe(high_risk[['id', 'address', 'risk_factor', 'xgb_prob', 'defensible_space_m']])

    st.markdown("---")
    st.markdown("**Note:** Probability based on XGBoost model using vegetation density and defensible space metrics extracted via SAM.")

