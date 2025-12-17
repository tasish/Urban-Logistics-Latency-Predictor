import streamlit as st
import pickle
import numpy as np
import time

# 1. Load Model & Scaler
# -------------------------------------------
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("⚠️ Error: model.pkl or scaler.pkl not found.")
    st.stop()

# 2. Page Configuration (Must be first)
# -------------------------------------------
st.set_page_config(
    page_title="Urban Logistics AI",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Dark Mode" friendly look
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar: Delivery Agent Profile
# -------------------------------------------
with st.sidebar:
    st.header("👤 Agent Profile")
    st.write("Configure the delivery partner details:")
    
    agent_age = st.slider("Agent Age", 18, 65, 30)
    agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5, 0.1)
    
    st.markdown("---")
    st.header("⚙️ System Settings")
    multiple_deliveries = st.radio("Concurrent Orders", [0, 1, 2, 3], index=0, horizontal=True)
    festival_input = st.toggle("🎉 Festival / Holiday Mode?", value=False)

# 4. Main Screen: Order Context
# -------------------------------------------
st.title("🚚 Urban Logistics AI")
st.markdown("### Last-Mile Delivery Latency Predictor")

# Create 3 columns for cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    st.info("📍 **Route Details**")
    distance = st.number_input("Distance (km)", 1, 50, 12)
    traffic_input = st.selectbox("Traffic Density", ['Low', 'Medium', 'High', 'Jam'], index=1)

with col2:
    st.warning("🌤️ **Environment**")
    weather_input = st.selectbox("Weather", ['Sunny', 'Cloudy', 'Windy', 'Fog', 'Stormy', 'Sandstorms'])
    # Helper to calculate day/month (hidden logic for demo)
    st.caption(f"Assuming typical order preparation time.")

with col3:
    st.success("🛵 **Fleet Info**")
    vehicle_input = st.selectbox("Vehicle Type", ['motorcycle', 'scooter', 'electric_scooter', 'bicycle'])
    st.write("") # Spacer
    st.write("") # Spacer

# 5. Mappings (Hidden Logic)
# -------------------------------------------
weather_map = {'Cloudy': 0, 'Fog': 1, 'Sandstorms': 2, 'Stormy': 3, 'Sunny': 4, 'Windy': 5}
traffic_map = {'High': 0, 'Jam': 1, 'Low': 2, 'Medium': 3}
vehicle_map = {'bicycle': 0, 'electric_scooter': 1, 'motorcycle': 2, 'scooter': 3}
festival_val = 1 if festival_input else 0

# 6. Build Input Array (29 Columns)
# -------------------------------------------
input_data = np.zeros(29)

# Map values to indices (Based on your X.columns)
input_data[0] = agent_age
input_data[1] = agent_rating
input_data[6] = weather_map[weather_input]
input_data[7] = traffic_map[traffic_input]
input_data[8] = 2   # Avg Vehicle condition
input_data[10] = vehicle_map[vehicle_input]
input_data[11] = multiple_deliveries
input_data[12] = festival_val
input_data[27] = 15 # Default Prep time
input_data[28] = distance

# 7. Prediction Section
# -------------------------------------------
st.markdown("---")
if st.button("🚀 Calculate ETA"):
    
    # Progress bar for visual effect
    with st.spinner('Analyzing route and traffic patterns...'):
        time.sleep(1) # Fake delay for effect
        
        # Predict
        scaled_input = scaler.transform(input_data.reshape(1, -1))
        prediction = int(model.predict(scaled_input)[0])

    # Dynamic Result Display
    st.markdown("### 🏁 Prediction Result")
    
    res_col1, res_col2, res_col3 = st.columns([1,2,1])
    
    with res_col2:
        # Determine Color
        if prediction < 25:
            lbl = "⚡ Fast Delivery"
            color = "green"
        elif prediction < 45:
            lbl = "⚠️ Standard Delivery"
            color = "orange"
        else:
            lbl = "🔴 High Delay"
            color = "red"
            
        st.metric(label=lbl, value=f"{prediction} mins", delta=f"{distance} km Route")

    # Expander for "Interviewer Mode" (Shows you know your stuff)
    with st.expander("🔍 See Model Inference Details"):
        st.json({
            "Algorithm": "XGBoost Regressor",
            "Input Features": 29,
            "Distance Impact": f"{distance} km",
            "Agent Efficiency": f"{agent_rating} Stars",
            "Raw Model Output": f"{model.predict(scaled_input)[0]:.4f}"
        })