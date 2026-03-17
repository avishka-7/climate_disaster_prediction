import streamlit as st
import requests
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
# LOAD MODEL
# =========================
model = joblib.load("xgboost_model.pkl")
features = joblib.load("model_features.pkl")

# =========================
# API KEY
# =========================
API_KEY = ""92c2e0509859c54d808577aac9ae09ea""

# =========================
# GET WEATHER DATA
# =========================
def get_weather(city):

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    
    if response.status_code != 200:
        return None, None, None

    data = response.json()

    weather = {
        "max_temperature": data["main"]["temp_max"],
        "min_temperature": data["main"]["temp_min"],
        "max_humidity": data["main"]["humidity"],
        "min_humidity": max(data["main"]["humidity"] - 10, 0),
        "wind_speed": data["wind"]["speed"] * 3.6,
        "pressure_surface_level": data["main"]["pressure"],
        "cloud_cover": data["clouds"]["all"],
        "visibility": data.get("visibility", 10000) / 1000,
        "uv_index": 7,  # approximation
        "solar_radiation": 500  # approximation
    }

    lat = data["coord"]["lat"]
    lon = data["coord"]["lon"]

    weather["latitude"] = lat
    weather["longitude"] = lon

    return weather, lat, lon


# =========================
# PREDICTION FUNCTION
# =========================
def predict_heatwave(city):

    weather, lat, lon = get_weather(city)

    if weather is None:
        return None, None, None, None, None, None

    df = pd.DataFrame([weather])
    df = df[features]

    probs = model.predict_proba(df)[0]
    heatwave_prob = probs[1] * 100

    prediction = "🔥 Heatwave" if heatwave_prob > 50 else "✅ No Heatwave"

    # Risk Level
    if heatwave_prob > 80:
        level = "Severe Risk 🔥"
    elif heatwave_prob > 60:
        level = "High Risk ⚠️"
    elif heatwave_prob > 40:
        level = "Moderate Risk"
    else:
        level = "Low Risk"

    return prediction, heatwave_prob, level, weather, lat, lon


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Heatwave Prediction", layout="wide")

st.title("🌍 Geospatial Heatwave Prediction System")

tabs = st.tabs([
    "🔍 Prediction",
    "🗺️ Risk Map",
    "🌆 Multi-City Monitor"
])

# =========================
# TAB 1: PREDICTION
# =========================
with tabs[0]:

    st.header("Heatwave Prediction")

    city = st.text_input("Enter City Name")

    if st.button("Predict"):

        with st.spinner("Fetching data & predicting..."):

            pred, prob, level, weather, lat, lon = predict_heatwave(city)

        if pred is None:
            st.error("❌ City not found. Please try again.")
        else:
            st.success("Prediction Complete!")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Result")
                st.write("City:", city)
                st.write("Prediction:", pred)
                st.write("Heatwave Probability:", round(prob,2), "%")
                st.write("Risk Level:", level)

            with col2:
                st.subheader("Weather Data")
                st.json(weather)

            # Probability chart
            prob_df = pd.DataFrame({
                "Class": ["No Heatwave", "Heatwave"],
                "Probability": [100-prob, prob]
            })

            st.subheader("Probability Breakdown")
            st.bar_chart(prob_df.set_index("Class"))

            # Feature importance
            st.subheader("Model Explanation")
            importance = model.feature_importances_

            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Feature"))


# =========================
# TAB 2: MAP
# =========================
with tabs[1]:

    st.header("Geospatial Heatwave Risk Map")

    city_map = st.text_input("Enter City")

    if st.button("Show Map"):

        pred, prob, level, weather, lat, lon = predict_heatwave(city_map)

        if pred is None:
            st.error("Invalid city")
        else:
            map_df = pd.DataFrame({
                "city":[city_map],
                "lat":[lat],
                "lon":[lon],
                "risk":[prob]
            })

            fig = px.scatter_mapbox(
                map_df,
                lat="lat",
                lon="lon",
                size="risk",
                color="risk",
                zoom=4,
                mapbox_style="open-street-map"
            )

            st.plotly_chart(fig)


# =========================
# TAB 3: MULTI-CITY
# =========================
with tabs[2]:

    st.header("Multi-City Heatwave Monitor")

    cities = [
        "Delhi","Mumbai","Chennai","Ahmedabad",
        "Bangalore","Kolkata","Hyderabad",
        "Pune","Jaipur","Lucknow"
    ]

    if st.button("Run Monitoring"):

        results = []

        with st.spinner("Analyzing cities..."):

            for city in cities:
                pred, prob, level, weather, lat, lon = predict_heatwave(city)

                if pred is not None:
                    results.append({
                        "City": city,
                        "Prediction": pred,
                        "Risk (%)": round(prob,2),
                        "Level": level
                    })

        df_monitor = pd.DataFrame(results)
        df_monitor = df_monitor.sort_values(by="Risk (%)", ascending=False)

        st.dataframe(df_monitor)
