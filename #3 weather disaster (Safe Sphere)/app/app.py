import streamlit as st
import joblib
import numpy as np
import pandas as pd
from weather_api import get_weather

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Safe Sphere | Disaster Risk Intelligence",
    # page_icon="🌍",
    layout="wide"
)

# ================= LOAD TRAINED MODEL =================
model = joblib.load("../model/multi_hazard_model.pkl")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
body { background-color: #0e1117; }
.metric-box {
    padding: 20px;
    border-radius: 12px;
    background: #161b22;
    text-align: center;
}
.metric-title {
    font-size: 16px;
    color: #9da5b4;
}
.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: white;
}
.risk-high { color: #ff4b4b; font-weight: bold; }
.risk-medium { color: #f7b731; font-weight: bold; }
.risk-low { color: #2ecc71; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("## Safe Sphere")
st.markdown(
    "**AI-Powered Environmental & Disaster Risk Forecasting Platform**  \n"
    "Live weather • Machine learning • Risk-based early warning"
)

st.divider()

# ================= INPUT =================
col1, col2 = st.columns([2, 1])

with col1:
    city = st.text_input("📍 Enter City Name", "Imphal")
    run = st.button(" Analyze Disaster Risk", use_container_width=True)

with col2:
    st.info(
        "This system uses **live weather data** and a **trained AI model** "
        "to predict disaster risk levels and support early decision-making."
    )

# ================= MAIN LOGIC =================
if run:
    with st.spinner("Fetching live weather & running AI model..."):
        weather = get_weather(city)

    if "error" in weather:
        st.error(f"Weather API Error: {weather['error']}")
        st.stop()

    rainfall = weather["rainfall"]
    temperature = weather["temperature"]
    humidity = weather["humidity"]

    # ================= FEATURE ENGINEERING =================
    rain_intensity = rainfall / 24
    heat_index = (temperature * humidity) / 100
    moisture_index = (rainfall * humidity) / 100
    soil_saturation = rainfall + moisture_index

    input_data = pd.DataFrame([{
        "rainfall_mm": rainfall,
        "temperature": temperature,
        "humidity": humidity,
        "rain_intensity": rain_intensity,
        "heat_index": heat_index,
        "moisture_index": moisture_index,
        "soil_saturation": soil_saturation
    }])

    # ================= PREDICTION =================
    preds = model.predict(input_data)[0]

    flood_risk, landslide_risk, heatwave_risk = preds

    def risk_label(val):
        if val == 2:
            return "HIGH", "risk-high"
        elif val == 1:
            return "MEDIUM", "risk-medium"
        return "LOW", "risk-low"

    flood_text, flood_class = risk_label(flood_risk)
    land_text, land_class = risk_label(landslide_risk)
    heat_text, heat_class = risk_label(heatwave_risk)

    # ================= WEATHER DISPLAY =================
    st.subheader(f"🌦 Live Weather — {city}")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Temperature</div>"
            f"<div class='metric-value'>{temperature} °C</div></div>",
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Humidity</div>"
            f"<div class='metric-value'>{humidity} %</div></div>",
            unsafe_allow_html=True
        )

    with m3:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Rainfall (1h)</div>"
            f"<div class='metric-value'>{rainfall} mm</div></div>",
            unsafe_allow_html=True
        )

    with m4:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Condition</div>"
            f"<div class='metric-value'>{weather['description']}</div></div>",
            unsafe_allow_html=True
        )

    st.divider()

    # ================= RISK OUTPUT =================
    st.subheader("⚠️ AI Disaster Risk Assessment")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Flood Risk</div>"
            f"<div class='metric-value {flood_class}'>{flood_text}</div></div>",
            unsafe_allow_html=True
        )

    with r2:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Landslide Risk</div>"
            f"<div class='metric-value {land_class}'>{land_text}</div></div>",
            unsafe_allow_html=True
        )

    with r3:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Heatwave Risk</div>"
            f"<div class='metric-value {heat_class}'>{heat_text}</div></div>",
            unsafe_allow_html=True
        )

    st.divider()

    # ================= MAP =================
    st.subheader("🗺 Geographic Visualization")

    map_df = pd.DataFrame({
        "lat": [weather["lat"]],
        "lon": [weather["lon"]]
    })

    st.map(map_df, zoom=7)

# ================= FOOTER =================
st.divider()
st.caption(
    "Safe Sphere • Prototype system • "
    "AI-assisted risk forecasting, not deterministic prediction"
)
