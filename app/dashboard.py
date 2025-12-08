import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import os
import altair as alt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Wildfire Survival Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Expert UI" feel
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #333;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stInfo {
        background-color: #f0f8ff;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    path = "app_data.csv"
    if not os.path.exists(path):
        path = "../app/app_data.csv"
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return None

df = load_data()

# --- HEADER & OVERVIEW ---
st.title("Palisades Wildfire Survival Model")

# --- METHODOLOGY MODAL ---
@st.dialog("Project Methodology")
def show_methodology():
    st.markdown("""
    ### Goal
    Predict whether individual homes would survive the 2025 Palisades Fire using only pre-fire satellite imagery. This model helps identify vulnerabilities *before* a disaster strikes.

    ### How it Works
    1.  **Satellite Imagery**: We downloaded high-resolution pre-fire images from Esri Wayback.
    2.  **Computer Vision (SAM)**: The **Segment Anything Model** extracted semantic masks to identify:
        *   🏠 **Structures** (Homes)
        *   🌲 **Vegetation** (Trees/Grass)
    3.  **Feature Engineering**: We calculated physics-based risk metrics (Defensible Space, Density, etc.).
    4.  **Machine Learning**: We trained models to predict burn probability:
        *   **XGBoost**: Gradient boosted trees on tabular physics features.
        *   **CNN (ResNet)**: A Convolutional Neural Network trained directly on image masks.
        
    **Note**: We initially trained an MLP (Multi-Layer Perceptron) with Focal Loss, but it did not perform well compared to XGBoost and CNN, so it was excluded from the final dashboard.

    ### Limitations
    *   **2D Imagery**: Does not account for slope/topography (a major fire factor).
    *   **Wind/Weather**: This is a static structural vulnerability model, not a real-time fire spread simulation.
    *   **Image Date**: Imagery is from late 2024; vegetation changes between then and the fire (Jan 2025) are not captured.
    """)

# --- SIDEBAR & NAVIGATION ---
st.sidebar.header("Control Panel")

if st.sidebar.button("📖 About this Project"):
    show_methodology()

st.sidebar.markdown("---")

if df is None:
    st.error("Data not found. Please run the data preparation notebooks.")
    st.stop()

# --- RISK FACTORS (FRONT PAGE) ---
st.subheader("Understanding Risk Factors")
col_edu1, col_edu2, col_edu3 = st.columns(3)

with col_edu1:
    st.info("""
    **Defensible Space**
    
    Distance to nearest vegetation.
    *   *Goal:* > 30ft (approx 10m).
    *   *Why:* Prevents direct flame contact.
    *   *Calculation:* Using SAM masks, we compute the distance transform from the structure mask to find the minimum distance to any tree pixel in the vegetation mask.
    """)

with col_edu2:
    st.info("""
    **Fuel Density**
    
    Ratio of flammable vegetation to lot size.
    *   *Why:* Denser vegetation burns hotter and longer.
    *   *Calculation:* From SAM masks, we sum tree pixels (weighted 1.5x) and divide by the total lot area (structure + tree + grass mask areas).
    """)

with col_edu3:
    st.info("""
    **Structure Compactness**
    
    Complexity of roof shape.
    *   *Why:* Complex roofs trap embers, increasing ignition risk.
    *   *Calculation:* From the SAM structure mask, we extract contours and compute (4π × area) / (perimeter²) to measure how circular/compact the footprint is.
    """)

st.markdown("---")

# --- MAIN CONTROLS (Top of Map) ---
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])

with col_ctrl1:
    view_mode = st.radio(
        "Layer Mode", 
        ["Risk Analysis", "Prediction vs Reality"],
        horizontal=True
    )

with col_ctrl2:
    # Model Selector
    available_models = ["XGBoost"]
    if 'cnn_prob' in df.columns: available_models.append("CNN (Visual)")
    
    selected_model_name = st.selectbox("Select Model", available_models)
    
    # Map name to column
    if "CNN" in selected_model_name:
        prob_col = "cnn_prob"
    else:
        prob_col = "xgb_prob"

with col_ctrl3:
    if view_mode == "Risk Analysis":
        selected_risks = st.multiselect(
            "Filter by Risk Factor",
            options=df['risk_factor'].unique(),
            default=df['risk_factor'].unique(),
            help="Select which types of vulnerabilities to display."
        )
        filtered_df = df[df['risk_factor'].isin(selected_risks)].copy()
    else:
        # Comparison logic
        threshold = st.slider("Probability Threshold", 0.0, 1.0, 0.5, 0.05, help="Cutoff for predicting 'Burned'")
        filtered_df = df.copy()
        
        # Handle NaNs in selected model (fallback to XGB or drop)
        if filtered_df[prob_col].isna().any():
            st.warning(f"⚠️ Some predictions missing for {selected_model_name}. Falling back to XGBoost where necessary.")
            filtered_df[prob_col] = filtered_df[prob_col].fillna(filtered_df['xgb_prob'])

        filtered_df['prediction'] = (filtered_df[prob_col] > threshold).astype(int)
        
        conditions = [
            (filtered_df['target'] == 1) & (filtered_df['prediction'] == 1),
            (filtered_df['target'] == 0) & (filtered_df['prediction'] == 0),
            (filtered_df['target'] == 0) & (filtered_df['prediction'] == 1),
            (filtered_df['target'] == 1) & (filtered_df['prediction'] == 0)
        ]
        choices = ["True Positive (Hit)", "True Negative (Safe)", "False Positive (False Alarm)", "False Negative (Miss)"]
        filtered_df['status'] = np.select(conditions, choices, default="Unknown")
        
        # Calculate performance metrics before filtering
        counts = filtered_df['status'].value_counts()
        st.markdown(f"##### 🎯 {selected_model_name} Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("True Positives", counts.get("True Positive (Hit)", 0), delta="Burned Correctly")
        m2.metric("True Negatives", counts.get("True Negative (Safe)", 0), delta="Saved Correctly")
        m3.metric("False Positives", counts.get("False Positive (False Alarm)", 0), delta_color="inverse", delta="Over-estimated")
        m4.metric("False Negatives", counts.get("False Negative (Miss)", 0), delta_color="inverse", delta="Under-estimated")
        st.markdown("---")
        
        selected_status = st.multiselect("Filter Outcome", options=choices, default=choices)
        filtered_df = filtered_df[filtered_df['status'].isin(selected_status)]

# --- MAP VISUALIZATION ---
# Logic for Colors
if view_mode == "Risk Analysis":
    # Handle NaNs for visualization
    viz_df = filtered_df.dropna(subset=[prob_col]).copy()
    
    # Gradient Red (High Risk) to Green (Low Risk)
    viz_df['color_r'] = (viz_df[prob_col] * 255).astype(int)
    viz_df['color_g'] = ((1 - viz_df[prob_col]) * 255).astype(int)
    viz_df['color_b'] = 0
    viz_df['radius'] = 15
    tooltip_html = f"<b>Risk:</b> {{risk_factor}}<br><b>{selected_model_name} Prob:</b> {{{prob_col}}}"
else:
    viz_df = filtered_df.copy()
    # Comparison Colors
    def get_color(status):
        if "Hit" in status: return [255, 0, 0]      # Red
        if "Safe" in status: return [0, 255, 0]     # Green
        if "False Alarm" in status: return [255, 165, 0] # Orange
        if "Miss" in status: return [148, 0, 211]   # Purple
        return [128, 128, 128]
    
    colors = viz_df['status'].apply(get_color)
    viz_df['color_r'] = colors.apply(lambda x: x[0])
    viz_df['color_g'] = colors.apply(lambda x: x[1])
    viz_df['color_b'] = colors.apply(lambda x: x[2])
    viz_df['radius'] = viz_df['status'].apply(lambda x: 25 if "False" in x else 12) # Highlight errors
    tooltip_html = f"<b>Status:</b> {{status}}<br><b>Reality:</b> {{target}}<br><b>{selected_model_name} Prob:</b> {{{prob_col}}}"

# PyDeck Map
layer = pdk.Layer(
    "ScatterplotLayer",
    viz_df,
    get_position=["lon", "lat"],
    get_fill_color=["color_r", "color_g", "color_b", 200],
    get_radius="radius",
    pickable=True,
    radius_min_pixels=3,
    radius_max_pixels=20,
)

view_state = pdk.ViewState(
    latitude=viz_df['lat'].mean(),
    longitude=viz_df['lon'].mean(),
    zoom=11.5,
    pitch=0,
)

# Dark Style (Removed Satellite)
r = pdk.Deck(
    map_style=pdk.map_styles.DARK, 
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"html": tooltip_html, "style": {"color": "white"}},
)

st.pydeck_chart(r)

# --- ANALYTICS DASHBOARD ---
st.subheader("📊 Analytics & Insights")

col_metrics1, col_metrics2 = st.columns([1, 1])

with col_metrics1:
    st.markdown(f"##### 🔥 Risk Distribution ({selected_model_name})")
    # Histogram of Probabilities
    if not viz_df.empty:
        chart = alt.Chart(viz_df).mark_bar().encode(
            x=alt.X(prob_col, bin=True, title="Burn Probability"),
            y='count()',
            color=alt.value("#ff4b4b")
        ).properties(height=200)
        st.altair_chart(chart, use_container_width=True)

with col_metrics2:
    st.markdown("##### 🌳 Vulnerability Factors")
    if not viz_df.empty:
        # Bar chart of Risk Factors
        risk_chart = alt.Chart(viz_df).mark_bar().encode(
            x=alt.X("count()", title="Count"),
            y=alt.Y("risk_factor", sort="-x", title="Primary Risk Factor"),
            color=alt.Color("risk_factor", legend=None)
        ).properties(height=200)
        st.altair_chart(risk_chart, use_container_width=True)
