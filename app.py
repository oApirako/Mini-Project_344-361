import streamlit as st
import pandas as pd
import joblib

# โหลดโมเดล
rf = joblib.load("rf_rain_model.pkl")
le_rain = joblib.load("le_rain.pkl")
le_raintoday = joblib.load("le_raintoday.pkl")

st.title("Rain Prediction App")
st.write("กรอกข้อมูลสภาพอากาศเพื่อพยากรณ์ว่าพรุ่งนี้จะมีฝนหรือไม่")

# กรอกข้อมูล
mintemp = st.number_input("MinTemp (°C)", -10.0, 40.0, 10.0)
maxtemp = st.number_input("MaxTemp (°C)", -10.0, 50.0, 25.0)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 0.0)
humidity3pm = st.number_input("Humidity 3 PM (%)", 0, 100, 57)
pressure3pm = st.number_input("Pressure 3 PM (hPa)", 900.0, 1100.0, 1017.6)
temp3pm = st.number_input("Temperature 3 PM (°C)", -10.0, 50.0, 21.8)
windgustspeed = st.number_input("Wind Gust Speed (km/h)", 0.0, 200.0, 35.0)
raintoday = st.selectbox("Did it rain today?", ["No", "Yes"])

# แปลงข้อมูล raintoday
raintoday_encoded = le_raintoday.transform([raintoday])[0]

# รวมข้อมูลเข้า DataFrame
user_input = pd.DataFrame([{
    'MinTemp': mintemp,
    'MaxTemp': maxtemp,
    'Rainfall': rainfall,
    'Humidity3pm': humidity3pm,
    'Pressure3pm': pressure3pm,
    'Temp3pm': temp3pm,
    'RainToday': raintoday_encoded,
    'WindGustSpeed': windgustspeed
}])

if st.button("Predict Rain Tomorrow"):
    #class
    prediction = rf.predict(user_input)[0]
    prediction_label = le_rain.inverse_transform([prediction])[0]

    #%
    prob_rain = rf.predict_proba(user_input)[0][1] * 100

    st.markdown(f"### ผลการพยากรณ์: **{'🌧️ ฝนตก' if prediction_label == 'Yes' else '☀️ ไม่ตก'}**")
    st.write(f"🔹 ความน่าจะเป็นที่ฝนจะตกพรุ่งนี้: **{prob_rain:.2f}%**") 
