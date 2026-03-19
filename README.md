# 📈 Stock Price Prediction using LSTM

🔗 **Live App:** https://nnpjcjynwjzdqfyaye4mgz.streamlit.app/

---

## 🚀 Overview

This project implements a **Stock Price Prediction System** using a **Long Short-Term Memory (LSTM)** neural network.  
It allows users to upload historical stock data and visualize:

- 📊 Closing price trends  
- 🔮 Model predictions vs actual values  
- 📅 Next-day predicted stock price  

The application is deployed using **Streamlit Cloud** for real-time interaction.

---

## 🧠 Why LSTM? (Mathematical Insight)

Stock prices are inherently **time-series data**, where future values depend on past observations.

Traditional models struggle with long-term dependencies. LSTM solves this using **gated mechanisms**.

### 🔑 Core LSTM Equations

At time step \( t \):

**Forget Gate:**
\[
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
\]

**Input Gate:**
\[
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
\]

**Candidate Cell State:**
\[
\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
\]

**Cell State Update:**
\[
C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t
\]

**Output Gate:**
\[
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
\]

**Hidden State:**
\[
h_t = o_t \cdot \tanh(C_t)
\]

---

## 📌 Interpretation

- **Forget gate** decides what past information to discard  
- **Input gate** decides what new information to store  
- **Cell state** acts as long-term memory  
- **Output gate** controls prediction  

👉 This architecture enables LSTM to capture **temporal dependencies and trends** in stock prices effectively.

---

## ⚙️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **ML Model:** LSTM (TensorFlow/Keras)  

**Libraries:**
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## 📂 Features

- 📁 Upload custom stock dataset (CSV)
- 📉 Visualize historical trends
- 🤖 Predict future stock prices
- 📊 Compare actual vs predicted values
- 🔮 Next-day price prediction

---

## 📊 Model Workflow

1. Data preprocessing (sorting, scaling using MinMaxScaler)
2. Sequence creation (last 60 timesteps)
3. LSTM training
4. Prediction on validation data
5. Visualization

---

## 🖥️ Run Locally

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
streamlit run app.py
