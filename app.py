import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

st.title("📈 Stock Price Prediction (LSTM)")

model_path = Path("saved_model.h5")
use_lstm_model = load_model is not None and model_path.exists()
model = load_model(model_path.as_posix()) if use_lstm_model else None

uploaded_file = st.file_uploader("Upload Stock CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    st.subheader("Closing Price Graph")
    st.line_chart(df["Close"])

    data = df.sort_index()
    new_dataset = data[['Close']]

    # Split
    train_data = new_dataset[:int(len(new_dataset)*0.7)]
    valid_data = new_dataset[int(len(new_dataset)*0.7):]

    valid_data = valid_data.copy()

    if use_lstm_model:
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(new_dataset)

        inputs = scaled_data[len(scaled_data) - len(valid_data) - 60:]

        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i-60:i, 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_price = model.predict(X_test, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_price)
        valid_data["Predictions"] = predicted_price
    else:
        st.info("TensorFlow model is unavailable in this environment. Using rolling-average fallback predictions.")
        fallback_pred = new_dataset["Close"].shift(1).rolling(window=5, min_periods=1).mean()
        valid_data["Predictions"] = fallback_pred.loc[valid_data.index].values

    st.subheader("Prediction vs Actual")

    fig = plt.figure(figsize=(10,5))
    plt.plot(train_data["Close"], label="Train")
    plt.plot(valid_data["Close"], label="Actual")
    plt.plot(valid_data["Predictions"], label="Predicted")
    plt.legend()
    st.pyplot(fig)

    # Next Day Prediction
    st.subheader("🔮 Next Day Prediction")

    if use_lstm_model:
        last_60 = new_dataset[-60:].values
        last_60 = scaler.transform(last_60)

        X_pred = np.array([last_60])
        X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))

        next_day = model.predict(X_pred, verbose=0)
        next_day = scaler.inverse_transform(next_day)
        next_day_value = float(next_day[0][0])
    else:
        next_day_value = float(new_dataset["Close"].tail(5).mean())

    st.success(f"Next Day Predicted Price: ₹ {next_day_value:.2f}")