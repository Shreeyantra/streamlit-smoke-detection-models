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

models = {}
for model_name, file_name in model_files.items():
    model_path = os.path.join(current_dir, file_name)
    if os.path.exists(model_path):
        model = load_model(model_path)
        if model is not None:
            models[model_name] = model

if len(models) == 0:
    st.error("No models available. Please retrain and save models with the individual-level features.")
    st.stop()

st.title("🔬 Fire Alarm Prediction")
selected_model_name = st.selectbox("🛠 Choose a Prediction Model:", list(models.keys()))
selected_model = models[selected_model_name]

expected_features = getattr(selected_model, "n_features_in_", None)

if expected_features is not None:
    st.write(f"✅ Model expects {expected_features} features.")
else:
    st.error("❌ The selected model does not have feature information. Check the training process.")
    st.stop()

input_feature_options = [
    "Temperature", "Humidity", "TVOC", "eCO2", "Raw H2", "Raw Ethanol",
    "Pressure", "PM1.0", "PM2.5", "PM10", "NC0.5", "NC1.0", "NC2.5", "CNT", "Fire Alarm"
]

# Collect inputs dynamically
input_values = []
for feature in input_feature_options:
    value = st.number_input(f"Enter value for {feature}", step=0.1)
    input_values.append(value)

# Match the input features to what the model expects
input_values = input_values[:expected_features]

if len(input_values) != expected_features:
    st.warning(f"⚠️ Feature mismatch: Model expects {expected_features} features, but got {len(input_values)}.")
    st.stop()

input_data = np.array([input_values], dtype=np.float64)

if st.button("🔍 Predict Fire Alarm Trigger"):
    try:
        ss = StandardScaler()
        input_data_scaled = ss.fit_transform(input_data)
        prediction = selected_model.predict(input_data_scaled)[0]
        st.success(f"🎯 Predicted Fire Alarm Likelihood: **{prediction:.2f}%**")
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")

