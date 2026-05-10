import streamlit as st
import pandas as pd
import joblib
import datetime


model = joblib.load('car_price_xgbmodel.pkl')
model_columns = joblib.load('model_columns.pkl')
st.title("Trợ Lý AI Định Giá Xe Ô Tô Cũ")
st.write("Nhập thông tin chiếc xe của bạn để AI dự đoán giá!")

brand = st.selectbox("Hãng xe", ['Ford', 'Toyota', 'BMW', 'Porsche', 'Audi', 'Mercedes-Benz'])
model_year = st.slider("Năm sản xuất", 2000, 2024, 2018)
milage = st.number_input("Số Km đã đi (Milage)", 0, 300000, 50000)
accident = st.selectbox("Lịch sử tai nạn", ['None reported', 'At least 1 accident or damage reported'])

if st.button('Dự đoán giá xe'):
    current_year = datetime.datetime.now().year
    car_age = current_year - model_year
    milage_per_year = milage / (car_age + 1)

    input_data = pd.DataFrame({
        'milage': [milage],
        'car_age': [car_age],
        'milage_per_year': [milage_per_year],
        'brand_' + brand: [1],
        'accident_' + accident: [1]
    })

    input_data = input_data.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(input_data)[0]
    
    st.success(f"Mức giá đề xuất (Độ chính xác 66%): ${prediction:,.0f}")
    st.balloons()
