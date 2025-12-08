import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import os

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

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("🔥 Palisades Wildfire Model")
st.sidebar.markdown("Predicting home survival probability using Satellite Imagery & ML.")

if df is not None:
    # --- VIEW MODE SELECTION ---
    view_mode = st.sidebar.radio(
        "View Mode", 
        ["Risk Analysis", "Prediction vs Reality"]
    )
    
    # --- FILTERS ---
    st.sidebar.header("Filters")
    
    if view_mode == "Risk Analysis":
        risk_filter = st.sidebar.multiselect(
            "Risk Factor", 
            options=df['risk_factor'].unique(),
            default=df['risk_factor'].unique()
        )
        filtered_df = df[df['risk_factor'].isin(risk_filter)].copy()
        
    else: # Comparison Mode
        # Calculate Status
        # Threshold Slider
        threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
        
        filtered_df = df.copy()
        filtered_df['prediction'] = (filtered_df['xgb_prob'] > threshold).astype(int)
        
        # Define Status
        conditions = [
            (filtered_df['target'] == 1) & (filtered_df['prediction'] == 1), # TP
            (filtered_df['target'] == 0) & (filtered_df['prediction'] == 0), # TN
            (filtered_df['target'] == 0) & (filtered_df['prediction'] == 1), # FP
            (filtered_df['target'] == 1) & (filtered_df['prediction'] == 0)  # FN
        ]
        choices = [
            "True Positive (Predicted Burn | Actual Burn)",
            "True Negative (Predicted Survive | Actual Survive)",
            "False Positive (Predicted Burn | Actual Survive)",
            "False Negative (Predicted Survive | Actual Burn)"
        ]
        filtered_df['status'] = np.select(conditions, choices, default="Unknown")
        
        status_filter = st.sidebar.multiselect(
            "Filter by Outcome",
            options=choices,
            default=choices
        )
        filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]

    # Show stats
    st.sidebar.markdown("---")
    st.sidebar.metric("Homes Displayed", len(filtered_df))
    
    if view_mode == "Comparison Mode":
        acc = (filtered_df['target'] == filtered_df['prediction']).mean()
        st.sidebar.metric("Accuracy (Selection)", f"{acc:.1%}")

# --- MAIN CONTENT ---
st.title("🏡 Wildfire Risk Map")

if df is None:
    st.error("Data not found. Please run 'notebooks/app_preparation.ipynb' to generate 'app_data.csv'.")
else:
    # --- VISUALIZATION LOGIC ---
    
    tooltip_html = ""
    
    if view_mode == "Risk Analysis":
        # Red for high risk, Green for low risk
        filtered_df['color_r'] = (filtered_df['xgb_prob'] * 255).astype(int)
        filtered_df['color_g'] = ((1 - filtered_df['xgb_prob']) * 255).astype(int)
        filtered_df['color_b'] = 0
        filtered_df['radius'] = 10
        
        tooltip_html = "<b>ID:</b> {id}<br><b>Risk:</b> {risk_factor}<br><b>Burn Prob:</b> {xgb_prob}"
        
    else: # Prediction vs Reality
        # Color Coding
        # TP (Red) - Correctly ID'd Hazard
        # TN (Green) - Correctly ID'd Safety
        # FP (Orange) - False Alarm
        # FN (Black/Purple) - Missed Hazard (Dangerous!)
        
        def get_color(status):
            if "True Positive" in status: return [255, 0, 0]    # Red
            if "True Negative" in status: return [0, 255, 0]    # Green
            if "False Positive" in status: return [255, 165, 0] # Orange
            if "False Negative" in status: return [128, 0, 128]# Purple
            return [128, 128, 128]
            
        colors = filtered_df['status'].apply(get_color)
        filtered_df['color_r'] = colors.apply(lambda x: x[0])
        filtered_df['color_g'] = colors.apply(lambda x: x[1])
        filtered_df['color_b'] = colors.apply(lambda x: x[2])
        
        # Emphasize Errors with larger radius
        filtered_df['radius'] = filtered_df['status'].apply(
            lambda x: 20 if "False" in x else 10
        )
        
        tooltip_html = """
        <b>ID:</b> {id}<br>
        <b>Status:</b> {status}<br>
        <b>Reality:</b> {target}<br>
        <b>Prediction:</b> {prediction}<br>
        <b>Prob:</b> {xgb_prob}
        """

    layer = pdk.Layer(
        "ScatterplotLayer",
        filtered_df,
        get_position=["lon", "lat"],
        get_fill_color=["color_r", "color_g", "color_b", 180],
        get_radius="radius",
        pickable=True,
        radius_min_pixels=3,
        radius_max_pixels=30,
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
        tooltip={"html": tooltip_html},
    )

    st.pydeck_chart(r)
    
    # --- LEGEND / METRICS ---
    if view_mode == "Prediction vs Reality":
        st.subheader("Comparison Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        tp = len(filtered_df[filtered_df['status'].str.contains("True Positive")])
        tn = len(filtered_df[filtered_df['status'].str.contains("True Negative")])
        fp = len(filtered_df[filtered_df['status'].str.contains("False Positive")])
        fn = len(filtered_df[filtered_df['status'].str.contains("False Negative")])
        
        col1.metric("True Positives (Burned)", tp)
        col2.metric("True Negatives (Survived)", tn)
        col3.metric("False Positives (False Alarm)", fp)
        col4.metric("False Negatives (Missed)", fn)
    
    # Data Table
    st.subheader("Data Inspector")
    st.dataframe(filtered_df[['id', 'address', 'target', 'xgb_prob', 'risk_factor']].head(50))
