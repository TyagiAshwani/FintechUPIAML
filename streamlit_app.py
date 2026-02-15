import streamlit as st
import pandas as pd
import numpy as np
import random
import string
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- HEADER ----------------
st.markdown("""
<style>
.header-container {display:flex;align-items:center;}
.logo {font-size:40px;font-weight:bold;color:#0A3D62;}
.title-text {font-size:32px;font-weight:bold;margin-left:15px;}
.subtitle {color:grey;margin-left:55px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-container">
    <div class="logo">🛡️</div>
    <div class="title-text">LayerGuard AI - Real-Time AML Solution</div>
</div>
<div class="subtitle">Intelligent Monitoring | Regulatory Ready | AI-Powered Risk Detection</div>
<hr>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Model Configuration")

sample1 = st.sidebar.number_input("Sample Transaction 1", min_value=0.0, value=5000.0)
sample2 = st.sidebar.number_input("Sample Transaction 2", min_value=0.0, value=6000.0)
sample3 = st.sidebar.number_input("Sample Transaction 3", min_value=0.0, value=4500.0)

std_percent = st.sidebar.slider("Std Deviation (% of Mean)", 1, 100, 20)
sensitivity = st.sidebar.slider("Risk Sensitivity Multiplier", 1.0, 5.0, 2.5)

# ---------------- ACCOUNT ID ----------------
def generate_account_id():
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    numbers = ''.join(random.choices(string.digits, k=2))
    return letters + numbers

# ---------------- MEAN & STD ----------------
mean_amount = np.mean([sample1, sample2, sample3])
std_amount = (std_percent / 100) * mean_amount

# ---------------- RISK SCORING FUNCTION ----------------
def calculate_risk(amount):
    z_score = abs((amount - mean_amount) / std_amount)
    risk_score = min(z_score * 20, 100)  # scale to 0-100
    return round(risk_score, 2)

# ---------------- LIVE TRANSACTION ----------------
if st.button("Generate Live Transaction"):

    new_std = 2 * std_amount
    new_amount = abs(np.random.normal(mean_amount, new_std))

    sender = generate_account_id()
    receiver = generate_account_id()

    risk_score = calculate_risk(new_amount)

    # Classification
    if risk_score > 70:
        category = "Suspicious"
        color = "red"
    elif risk_score > 50:
        category = "Potential"
        color = "orange"
    else:
        category = "Normal"
        color = "green"

    st.subheader("Current Transaction Analysis")
    col1, col2, col3 = st.columns(3)

    col1.metric("Amount", round(new_amount,2))
    col2.metric("Risk Score", risk_score)
    col3.markdown(f"<h4 style='color:{color}'>{category}</h4>", unsafe_allow_html=True)

    # Store only flagged
    if "alerts" not in st.session_state:
        st.session_state.alerts = pd.DataFrame(
            columns=["Sender","Receiver","Amount","Risk Score","Category"]
        )

    if category in ["Suspicious","Potential"]:
        new_row = pd.DataFrame({
            "Sender":[sender],
            "Receiver":[receiver],
            "Amount":[round(new_amount,2)],
            "Risk Score":[risk_score],
            "Category":[category]
        })
        st.session_state.alerts = pd.concat(
            [st.session_state.alerts,new_row], ignore_index=True
        )

# ---------------- ALERT TABLE ----------------
st.markdown("---")
st.subheader("Flagged Transactions (Suspicious & Potential Only)")

if "alerts" in st.session_state:
    st.dataframe(st.session_state.alerts)
else:
    st.write("No flagged transactions yet.")

# ---------------- PERFORMANCE SECTION ----------------
st.markdown("---")
st.header("Our Performance")

if st.button("Show Performance"):

    # Generate test dataset
    test_data = pd.DataFrame({
        'amount': np.abs(np.random.normal(mean_amount, std_amount, 1000))
    })

    # Inject synthetic anomalies (5%)
    anomaly_indices = np.random.choice(1000, 50, replace=False)
    test_data.loc[anomaly_indices, 'amount'] *= 3

    # True labels
    y_true = np.zeros(1000)
    y_true[anomaly_indices] = 1

    # ---------------- MODEL RISK SCORING ----------------
    true_risk = np.abs((test_data['amount'] - mean_amount) / std_amount)

    # Sensitivity influences model aggressiveness
    sensitivity_factor = sensitivity / 5  # normalize roughly
    noise = np.random.normal(0, 0.3, len(true_risk))

    model_risk = true_risk * sensitivity_factor * (1 + noise)
    model_scaled = (model_risk / max(model_risk)) * 100

    # Classification threshold based on sensitivity
    threshold = 60 - (sensitivity * 1.5)
    y_pred = (model_scaled > threshold).astype(int)

    # ---------------- METRICS ----------------
    TP = sum((y_true == 1) & (y_pred == 1))
    FP = sum((y_true == 0) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FN = sum((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    correlation = np.corrcoef(true_risk, model_scaled)[0,1]

    # ---------------- DRIFT DETECTION ----------------
    drift_mean = abs(test_data['amount'].mean() - mean_amount)
    drift_score = drift_mean / mean_amount

    # ---------------- DISPLAY METRICS ----------------
    st.subheader("Model Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{round(precision*100,2)} %")
    col2.metric("Recall", f"{round(recall*100,2)} %")
    col3.metric("F1 Score", f"{round(f1*100,2)} %")

    col4, col5 = st.columns(2)
    col4.metric("Risk Correlation", f"{round(correlation*100,2)} %")
    col5.metric("Drift Score", f"{round(drift_score*100,2)} %")

    st.markdown("Sensitivity directly affects detection threshold and aggressiveness.")

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("Confusion Matrix")

    cm_df = pd.DataFrame(
        [[TP, FP],
         [FN, TN]],
        columns=["Predicted Fraud", "Predicted Normal"],
        index=["Actual Fraud", "Actual Normal"]
    )

    st.dataframe(cm_df)

    # ---------------- 3D VISUALIZATION ----------------
    st.subheader("3D Risk Distribution Map")

    fig = go.Figure(data=[go.Scatter3d(
        x=test_data['amount'][:200],
        y=true_risk[:200],
        z=model_scaled[:200],
        mode='markers',
        marker=dict(
            size=5,
            color=model_scaled[:200],
            colorscale='RdYlGn_r',
            colorbar=dict(title="Risk Score")
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='Transaction Amount',
            yaxis_title='True Risk Intensity',
            zaxis_title='Model Risk Score'
        ),
        height=700
    )

    st.plotly_chart(fig)
