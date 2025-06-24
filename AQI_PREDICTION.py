import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
with gzip.open("aqi_preediction_.pkl.gz", "rb") as f:
    model = joblib.load(f)

# Load dataset
df = pd.read_csv("C://Users//USER//Downloads//folder//city_day.csv")  # Replace with actual path

st.title("\U0001F32B️ Air Quality Monitoring & AQI Bucket Prediction")

# -------------------------------
# 1. Data Overview
st.header("\U0001F4CA Dataset Overview")
st.write(df.head())

if st.checkbox("Show Summary Statistics"):
    st.write(df.describe())

# -------------------------------
# 2. Correlation Heatmap
st.subheader("\U0001F4CC Correlation Heatmap")
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Error generating heatmap: {e}")

# -------------------------------
# 3. PM2.5 vs AQI Scatter
if "PM2.5" in df.columns and "AQI" in df.columns and "AQI_Bucket" in df.columns:
    st.subheader("\U0001F4C8 PM2.5 vs AQI")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="PM2.5", y="AQI", hue="AQI_Bucket", palette="Set1", ax=ax2)
    st.pyplot(fig2)
else:
    st.warning("Required columns for scatter plot ('PM2.5', 'AQI', 'AQI_Bucket') not found in dataset.")

# -------------------------------
# 4. Feature Importance
st.subheader("\U0001F31F Feature Importance")

features = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene",
            "Xylene", "AQI", "Year", "Month", "Day", "Hour", "Location Encoded", "Season Encoded",
            "Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Wind Direction (°)", "Dew Point (°C)",
            "Pressure (hPa)", "Rainfall (mm)", "Solar Radiation", "Cloud Cover (%)", "Visibility (km)",
            "PM1", "CO2", "CH4", "Lead", "Arsenic", "Nickel", "Ammonia", "Methane",
            "Ethylene", "Propylene", "Butadiene", "Formaldehyde", "Acetaldehyde"]

try:
    if hasattr(model, "feature_importances_") and len(model.feature_importances_) == len(features):
        importance = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(by="Importance", ascending=False)
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax3)
        st.pyplot(fig3)
    else:
        st.warning("\u26a0\ufe0f Model does not support feature importances or feature count mismatch.")
except Exception as e:
    st.warning(f"\u274c Error computing feature importances: {e}")

# -------------------------------
# 5. Additional Plots
st.header("\U0001F3A8 More Visualizations")

# AQI Bucket Pie Chart
if "AQI_Bucket" in df.columns:
    st.subheader("\U0001F4E6 AQI Bucket Distribution")
    aqi_counts = df["AQI_Bucket"].value_counts()
    fig4, ax4 = plt.subplots()
    aqi_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, shadow=True, ax=ax4)
    ax4.set_ylabel("")
    ax4.set_title("Distribution of AQI Buckets")
    st.pyplot(fig4)

# PM2.5 over Time
if "Date" in df.columns and "PM2.5" in df.columns:
    st.subheader("\U0001F4C8 PM2.5 Trend Over Time")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    daily_avg = df.groupby(df['Date'].dt.date)["PM2.5"].mean().reset_index()
    fig5, ax5 = plt.subplots()
    sns.lineplot(data=daily_avg, x="Date", y="PM2.5", ax=ax5)
    ax5.set_title("Daily Average PM2.5 Levels")
    plt.xticks(rotation=45)
    st.pyplot(fig5)

# City-wise AQI Comparison
if "City" in df.columns and "AQI" in df.columns:
    st.subheader("\U0001F3D9\ufe0f City-Wise AQI Comparison")
    city_avg_aqi = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)
    fig6, ax6 = plt.subplots()
    city_avg_aqi.plot(kind='barh', color='skyblue', ax=ax6)
    ax6.set_title("Top 10 Cities by Average AQI")
    st.pyplot(fig6)

# AQI vs Wind Speed
if "AQI" in df.columns and "Wind Speed (km/h)" in df.columns:
    st.subheader("\U0001F32C\ufe0f AQI vs Wind Speed")
    fig7, ax7 = plt.subplots()
    sns.scatterplot(data=df, x="Wind Speed (km/h)", y="AQI", hue="AQI_Bucket", palette="coolwarm", alpha=0.6, ax=ax7)
    ax7.set_title("Effect of Wind Speed on AQI")
    st.pyplot(fig7)

# AQI by Month
if "Month" in df.columns and "AQI" in df.columns:
    st.subheader("\U0001F4C5 AQI by Month")
    fig8, ax8 = plt.subplots()
    sns.boxplot(data=df, x="Month", y="AQI", ax=ax8)
    ax8.set_title("AQI Distribution Across Months")
    st.pyplot(fig8)

# -------------------------------
# 6. AQI Prediction
st.header("\U0001F3AF Predict AQI Category")
st.markdown("#### Adjust values if needed, or use the defaults to test a prediction:")

inputs = []
def_input = [80.0, 120.0, 20.0, 30.0, 40.0, 15.0, 0.7, 10.0, 25.0, 1.2, 2.5, 0.9, 150.0,
             2024, 6, 21, 12, 2, 1, 32.0, 60.0, 10.0, 180.0, 24.0, 1010.0, 2.0, 400.0, 30.0,
             5.0, 60.0, 400.0, 1.8, 0.05, 0.02, 0.01, 5.0, 1.5, 0.3, 0.4, 0.2, 0.05, 0.07]

for i, feat in enumerate(features):
    val = st.number_input(feat, value=def_input[i])
    inputs.append(val)

if st.button("Predict AQI Bucket"):
    input_array = np.array([inputs])
    prediction = model.predict(input_array)[0]
    category_map = {0: "Good", 1: "Satisfactory", 2: "Moderate", 3: "Poor", 4: "Very Poor", 5: "Severe"}
    st.success(f"\u2705 Predicted AQI Bucket: **{prediction} - {category_map.get(prediction, 'Unknown')}**")
