import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Kualitas Udara (NO₂)", layout="centered")

# -------------------------------------------------------------
# Fungsi bantu
# -------------------------------------------------------------
def generate_future_dates(last_date, days):
    return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')

def categorize_no2(value):
    if value <= 40:
        return "Baik"
    elif 41 <= value <= 100:
        return "Sedang"
    else:
        return "Buruk"

# -------------------------------------------------------------
# Tampilan aplikasi
# -------------------------------------------------------------
st.title("Prediksi Kualitas Udara Harian Berdasarkan NO₂")
st.write("Aplikasi ini memprediksi konsentrasi NO₂ untuk 1–7 hari ke depan menggunakan model **KNN Time Series Forecasting**.")
st.markdown("---")

uploaded_file = st.file_uploader("Unggah data CSV (harus memiliki kolom 'date' dan 'no2')", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    st.subheader("Data Awal")
    st.dataframe(df.tail(10))

    # Normalisasi
    scaler = MinMaxScaler()
    df['no2_scaled'] = scaler.fit_transform(df[['no2']])

    # Siapkan data train
    X, y = [], []
    for i in range(len(df)-3):
        X.append(df['no2_scaled'].iloc[i:i+3].values)
        y.append(df['no2_scaled'].iloc[i+3])
    X, y = np.array(X), np.array(y)

    # Latih model KNN
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X, y)

    # Input hari ke depan
    n_days = st.slider("Jumlah hari yang ingin diprediksi:", 1, 7, 3)
    last_values = df['no2_scaled'].iloc[-3:].values.reshape(1, -1)

    preds = []
    for _ in range(n_days):
        next_pred = model.predict(last_values)[0]
        preds.append(next_pred)
        last_values = np.roll(last_values, -1)
        last_values[0, -1] = next_pred

    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    # Tanggal prediksi
    future_dates = generate_future_dates(df['date'].iloc[-1], n_days)
    result_df = pd.DataFrame({
        'Tanggal': future_dates,
        'Prediksi_NO2': preds_inv,
        'Kategori': [categorize_no2(v) for v in preds_inv]
    })

    st.subheader("Hasil Prediksi")
    st.dataframe(result_df)

    # Grafik
    st.subheader("Grafik Konsentrasi NO₂")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df['date'], df['no2'], label="Data Aktual", marker='o')
    ax.plot(result_df['Tanggal'], result_df['Prediksi_NO2'], label="Prediksi", marker='x', linestyle='--')
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Konsentrasi NO₂ (µg/m³)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Rangkuman kategori
    st.subheader("Rangkuman Kualitas Udara")
    counts = result_df['Kategori'].value_counts()
    st.write(counts)

else:
    st.info("Silakan unggah file CSV yang berisi kolom 'date' dan 'no2' untuk memulai prediksi.")
