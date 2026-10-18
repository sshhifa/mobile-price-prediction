
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Mobile Price Prediction",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Mobile Price Prediction System")
st.write("Enter mobile specifications to predict its price in INR.")

# Load trained model and feature information
model = joblib.load("mobile_price_model.pkl")
feature_info = joblib.load("feature_info.pkl")

st.subheader("📋 Mobile Specifications")

brand = st.text_input("Smartphone Brand", "samsung")
processor = st.text_input("Processor Name", "exynos 1480")
processor_brand = st.text_input("Processor Brand", "exynos")

col1, col2 = st.columns(2)

with col1:
    core_count = st.number_input("CPU Cores", 1, 16, 8)
    clock_speed = st.number_input("Clock Speed (GHz)", 0.1, 6.0, 2.7)
    ram = st.number_input("RAM (GB)", 1, 64, 8)
    storage = st.number_input("Storage (GB)", 8, 2048, 256)
    display = st.number_input("Display Size (inches)", 3.0, 10.0, 6.74)
    res_width = st.number_input("Resolution Width (px)", 100, 5000, 1080)
    res_height = st.number_input("Resolution Height (px)", 100, 6000, 2340)
    refresh_rate = st.number_input("Refresh Rate (Hz)", 30, 240, 120)
    battery = st.number_input("Battery (mAh)", 500, 15000, 5000)

with col2:
    has_5g = st.checkbox("5G Support", value=True)
    has_nfc = st.checkbox("NFC Support", value=False)
    has_ir = st.checkbox("IR Blaster", value=False)
    fast_charging = st.checkbox("Fast Charging", value=True)
    charging_watt = st.number_input("Charging Watt", 0.0, 300.0, 45.0)
    rear_count = st.number_input("Rear Camera Count", 1, 10, 3)
    front_count = st.number_input("Front Camera Count", 1, 10, 1)
    rear_mp = st.number_input("Rear Main Camera (MP)", 1.0, 300.0, 50.0)
    front_mp = st.number_input("Front Main Camera (MP)", 1.0, 200.0, 12.0)
    os_name = st.selectbox("Operating System", ["android", "ios", "other"])
    memory_supported = st.selectbox(
        "Memory Card Supported",
        ["Unknown", "True", "False"]
    )
    memory_type = st.text_input("Memory Card Type", "Unknown")

# Prepare all 24 features
input_data = {
    "smartphone_brand": brand,
    "processor_name": processor,
    "processor_brand": processor_brand,
    "core_count": core_count,
    "clock_speed_ghz": clock_speed,
    "ram_gb": ram,
    "storage_gb": storage,
    "has_5g": has_5g,
    "has_nfc": has_nfc,
    "has_ir_blaster": has_ir,
    "display_inches": display,
    "res_width_px": res_width,
    "res_height_px": res_height,
    "refresh_rate_hz": refresh_rate,
    "battery_mah": battery,
    "fast_charging": fast_charging,
    "charging_watt": charging_watt,
    "rear_camera_count": rear_count,
    "front_camera_count": front_count,
    "rear_camera_main_mp": rear_mp,
    "front_camera_main_mp": front_mp,
    "os_name": os_name,
    "memory_card_supported": memory_supported,
    "memory_card_type": memory_type
}

input_df = pd.DataFrame([input_data])

# Keep exact feature order used during training
input_df = input_df[feature_info["columns"]]

# Apply the same categorical conversion used when saving model
for col in feature_info["categorical_cols"]:
    input_df[col] = input_df[col].fillna("Unknown").astype(str)

if st.button("🔮 Predict Price", type="primary"):
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Mobile Price: ₹{prediction:,.2f}")
    st.caption("This is an estimated price based on the trained ML model.")
