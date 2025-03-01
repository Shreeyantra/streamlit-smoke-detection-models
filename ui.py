import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Get the directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Function to load a model safely using joblib
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {model_path}. {str(e)}")
        return None

model_files = {

    "ADB": "ADB.pkl",
    "BC": "BC.pkl",
    "DC": "DC.pkl",
    "ETC": "ETC.pkl",
    "GBC": "GBC.pkl",
    "GNB": "GNB.pkl",
    "KNM": "KNM.pkl",
    "LR": "LR.pkl",
    "RFC": "RFC.pkl",
    "SGD": "SGD.pkl",
    "SVC": "SVC.pkl",
    "ensemble": "ensemble.pkl",
    "stacking": "stacking.pkl"

}

# Load models dynamically
models = {}
for model_name, file_name in model_files.items():
    model_path = os.path.join(current_dir, file_name)
    if os.path.exists(model_path):
        model = load_model(model_path)
        if model is not None:
            models[model_name] = model
    else:
        st.warning(f"⚠️ Model '{model_name}' not found! Skipping...")

st.title("🔬 Lead Conversion Prediction")
st.markdown("### **Select a model and enter your details to predict the likelihood of conversion**")

if len(models) == 0:
    st.error("No models available. Please retrain and save models with the individual-level features.")
    st.stop()

selected_model_name = st.selectbox("🛠 Choose a Prediction Model:", list(models.keys()))
selected_model = models[selected_model_name]

if not hasattr(selected_model, "predict"):
    st.error("❌ Error: The selected model is not valid. Please choose another model.")
    st.stop()

st.write("### **📊 Enter Your Personal & Lead Details**")

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("🌡️ Temperature (C)", min_value=-50.0, max_value=50.0, step=0.1, help="Current temperature in Celsius.")
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, help="Current humidity in percentage.")
    tvoc = st.number_input("🌿 TVOC (ppb)", min_value=0, max_value=1000, step=1, help="Total Volatile Organic Compounds in ppb.")
    eCO2 = st.number_input("⚡ eCO2 (ppm)", min_value=0, max_value=1000, step=1, help="Equivalent CO2 concentration in ppm.")
    raw_h2 = st.number_input("🔋 Raw H2", min_value=0, max_value=50000, step=1, help="Raw hydrogen sensor reading.")
    raw_ethanol = st.number_input("🥃 Raw Ethanol", min_value=0, max_value=50000, step=1, help="Raw ethanol sensor reading.")
    pressure = st.number_input("🌬️ Pressure (hPa)", min_value=800.0, max_value=1100.0, step=0.1, help="Atmospheric pressure in hPa.")
    pm1 = st.number_input("🧳 PM1.0", min_value=0.0, max_value=1000.0, step=0.1, help="PM1.0 particulate matter concentration.")
    pm2_5 = st.number_input("🧳 PM2.5", min_value=0.0, max_value=1000.0, step=0.1, help="PM2.5 particulate matter concentration.")
    pm10 = st.number_input("🧳 PM10", min_value=0.0, max_value=1000.0, step=0.1, help="PM10 particulate matter concentration.")
    nc0_5 = st.number_input("🌱 NC0.5", min_value=0.0, max_value=1000.0, step=0.1, help="NC0.5 particulate matter concentration.")
    
with col2:
    nc1_0 = st.number_input("🌱 NC1.0", min_value=0.0, max_value=1000.0, step=0.1, help="NC1.0 particulate matter concentration.")
    nc2_5 = st.number_input("🌱 NC2.5", min_value=0.0, max_value=1000.0, step=0.1, help="NC2.5 particulate matter concentration.")
    cnt = st.number_input("📊 CNT", min_value=0, max_value=10000, step=1, help="CNT sensor count.")
    fire_alarm = st.number_input("🔥 Fire Alarm", min_value=0, max_value=1, step=1, help="Fire alarm status (0: no fire, 1: fire).")

# Feature preprocessing (one-hot encoding for categorical data)
# In this case, we'll skip the categorical mapping since all input features are numeric.
input_data = np.array([[
    temperature,
    humidity,
    tvoc,
    eCO2,
    raw_h2,
    raw_ethanol,
    pressure,
    pm1,
    pm2_5,
    pm10,
    nc0_5,
    nc1_0,
    nc2_5,
    cnt,
    fire_alarm
]], dtype=np.float64)

# Check that the model expects the same number of features
expected_features = getattr(selected_model, "n_features_in_", input_data.shape[1])
if input_data.shape[1] != expected_features:
    st.error(
        f"❌ Feature Mismatch: The selected model expects {expected_features} features, "
        f"but got {input_data.shape[1]}. Please retrain the model with the individual-level features."
    )
    st.stop()

if st.button("🔍 Predict Fire Alarm Trigger"):
    try:
        prediction = selected_model.predict(input_data)[0]
        fire_alarm_probability = max(0, min(100, prediction))
        st.success(f"🎯 Predicted Fire Alarm Likelihood: **{fire_alarm_probability:.2f}%**")
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
