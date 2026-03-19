import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.title("📈 Stock Price Prediction (LSTM)")

# Upload CSV
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

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_train = scaler.fit_transform(train_data)

    x_train, y_train = [], []

    for i in range(60, len(scaled_train)):
        x_train.append(scaled_train[i-60:i, 0])
        y_train.append(scaled_train[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Train Model
    st.subheader("Training Model...")

    model = Sequential()
    model.add(LSTM(units=150, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=100))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=5, batch_size=2, verbose=0)

    # Prepare test data
    inputs = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    valid_data = valid_data.copy()
    valid_data["Predictions"] = predicted_price

    st.subheader("Prediction vs Actual")

    fig = plt.figure(figsize=(10,5))
    plt.plot(train_data["Close"], label="Train")
    plt.plot(valid_data["Close"], label="Actual")
    plt.plot(valid_data["Predictions"], label="Predicted")
    plt.legend()
    st.pyplot(fig)

    # Next Day Prediction
    st.subheader(" Next Day Prediction")

    last_60 = new_dataset[-60:].values
    last_60 = scaler.transform(last_60)

    X_pred = []
    X_pred.append(last_60)
    X_pred = np.array(X_pred)
    X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))

    next_day = model.predict(X_pred)
    next_day = scaler.inverse_transform(next_day)

    st.success(f"Next Day Predicted Price: ₹ {next_day[0][0]:.2f}")