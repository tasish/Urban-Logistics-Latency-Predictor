import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from PIL import Image
import os
import src.config as config

# Page Config
st.set_page_config(
    page_title="Urban Logistics AI",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold;}
    .reportview-container .main .block-container{ padding-top: 2rem; }
    h1, h2, h3 { color: #FAFAFA; }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    } 
    </style>
    """, unsafe_allow_html=True)

# Helper: Load Model
@st.cache_resource
def load_artifacts():
    try:
        with open(config.MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_artifacts()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=80)
    st.title("Urban Logistics AI")
    st.caption("Delivery time prediction")
    st.markdown("---")
    st.info("This project uses **XGBoost (R²=0.84)** to predict delivery latency with traffic-aware geospatial features.")
    
    st.markdown("### 👨‍💻 Project By")
    st.text("Thatikonda Asish")
    st.markdown("[GitHub Repo](https://github.com/tasish/Urban-Logistics-Latency-Predictor)")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Predictor", "📊 Analytics", "🧠 Model Details"])

# --- TAB 1: PREDICTOR ---
with tab1:
    st.header("⏱️ Real-time Delivery Estimate")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Order Details")
        
        # 1. Logistics Inputs
        distance = st.slider("📍 Distance (km)", 1.0, 30.0, 8.5)
        traffic_label = st.select_slider("🚦 Traffic Density", options=['Low', 'Medium', 'High', 'Jam'], value='Medium')
        weather_label = st.selectbox("⛅ Weather", options=list(config.WEATHER_MAP.keys())[6:]) # Use clean keys
        
        # 2. Agent Inputs
        c1, c2 = st.columns(2)
        agent_age = c1.number_input("Agent Age", 18, 60, 28)
        agent_rating = c2.number_input("Agent Rating", 1.0, 5.0, 4.6, 0.1)
        
        # 3. Vehicle
        vehicle_label = st.selectbox("🛵 Vehicle Type", options=list(config.VEHICLE_MAP.keys()))
        
        # 4. Others
        multiple = st.checkbox("Multiple Deliveries?", value=False)
        fast_prep = st.checkbox("Fast Prep Time (<10m)?", value=False)
        prep_time = 10 if fast_prep else 15

    with col2:
        st.subheader("🏁 Prediction")
        
        if st.button("Calculate ETA", type="primary"):
            if model is None:
                st.error("Model not found! Run training pipeline first.")
            else:
                with st.spinner("Processing geospatial & traffic data..."):
                    time.sleep(0.5) # UX Delay
                    
                    # Encode Inputs
                    weather_val = config.WEATHER_MAP.get(weather_label, 0)
                    traffic_val = config.TRAFFIC_MAP.get(traffic_label, 3)
                    vehicle_val = config.VEHICLE_MAP.get(vehicle_label, 2)
                    traffic_factor = config.TRAFFIC_FACTOR_MAP.get(traffic_label, 1.0)
                    city_val = 0 # Default Metropolitian
                    festival_val = 0
                    
                    # Create Input Array (Must match training order)
                    # ['Delivery_person_Age', 'Delivery_person_Ratings', 'Weather_Encoded', 
                    # 'Traffic_Encoded', 'Vehicle_condition', 'Vehicle_Encoded', 'multiple_deliveries', 
                    # 'Festival_Encoded', 'City_Encoded', 'Prep_Time', 'Distance_km', 'Traffic_Adj_Dist']
                    
                    input_data = np.array([[
                        agent_age,
                        agent_rating,
                        weather_val,
                        traffic_val,
                        1, # Vehicle Condition (Avg)
                        vehicle_val,
                        1 if multiple else 0,
                        festival_val,
                        city_val,
                        prep_time,
                        distance,
                        distance * traffic_factor # Interaction Feature
                    ]])
                    
                    # Scale
                    input_scaled = scaler.transform(input_data)
                    
                    # Predict
                    pred = model.predict(input_scaled)[0]
                    
                    # Display Result
                    st.metric(label="Estimated Latency", value=f"{int(pred)} min")
                    
                    # Contextual Message
                    if pred < 20:
                        st.success("⚡ Super Fast! Optimal traffic conditions.")
                    elif pred < 40:
                        st.info("🚗 Standard delivery time.")
                    else:
                        st.warning("🐌 Delays expected due to traffic/distance.")

                    # Feature Impact Explanation
                    st.markdown("#### Why this result?")
                    impact = (distance * traffic_factor * 1.5) + ((5 - agent_rating) * 2)
                    st.progress(min(int(impact * 2), 100), text="Route Complexity Score")

# --- TAB 2: ANALYTICS ---
with tab2:
    st.header("📈 Operational Analytics")
    
    st.subheader("🔥 Delivery Hotspots")
    # Embed HTML Map
    try:
        with open("assets/delivery_hotspots.html", "r", encoding='utf-8') as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=500, scrolling=True)
    except FileNotFoundError:
        st.warning("Map artifact not found. Run insights pipeline.")
        
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.image("assets/prep_time_dist.png", caption="Prep Time Analysis")
    with c2:
        st.image("assets/performance_by_age.png", caption="Age vs Performance")

# --- TAB 3: MODEL DETAILS ---
with tab3:
    st.header("🧠 Data Science & ML Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Learning Curve")
        st.image("assets/learning_curve.png", caption="Bias-Variance Tradeoff")
        st.markdown("""
        **Analysis:**
        * Shows convergence between training and validation scores.
        * Gap indicates low variance (good generalization).
        """)
        
    with col2:
        st.markdown("### Algorithm Comparison")
        st.image("assets/model_comparison.png", caption="XGBoost vs Random Forest")
        st.markdown("""
        **Findings:**
        * XGBoost consistently outperforms Bagging methods.
        * Lower Error spread indicates stability.
        """)