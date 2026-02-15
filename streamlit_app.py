import streamlit as st
import pandas as pd
import numpy as np
import random
import string
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
import streamlit as st
import pandas as pd
import numpy as np
import random
import string
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(layout="wide")

# ---------------- LOGO DESIGN ----------------
st.markdown("""
<style>
.header-container {
    display: flex;
    align-items: center;
}
.logo {
    font-size:40px;
    font-weight:bold;
    color:#0A3D62;
}
.title-text {
    font-size:32px;
    font-weight:bold;
    margin-left:15px;
}
.subtitle {
    color:grey;
    margin-left:55px;
}
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
# -------------------------
# USER INPUT SECTION
# -------------------------
st.sidebar.header("Model Configuration")

sample1 = st.sidebar.number_input("Sample Transaction 1", value=5000.0)
sample2 = st.sidebar.number_input("Sample Transaction 2", value=6000.0)
sample3 = st.sidebar.number_input("Sample Transaction 3", value=4500.0)

std_percent = st.sidebar.slider("Std Deviation % of Mean", 1, 100, 20)
sensitivity = st.sidebar.slider("Market Sensitivity (Contamination %)", 1, 20, 5)

# -------------------------
# ACCOUNT ID GENERATOR
# -------------------------
def generate_account_id():
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    numbers = ''.join(random.choices(string.digits, k=2))
    return letters + numbers

# -------------------------
# MODEL TRAINING FUNCTION
# -------------------------
def train_model(mean_amt, std_amt, contamination):
    data = pd.DataFrame({
        "amount": np.abs(np.random.normal(mean_amt, std_amt, 1000)),
        "sender": [generate_account_id() for _ in range(1000)],
        "receiver": [generate_account_id() for _ in range(1000)]
    })
    
    model = IsolationForest(contamination=contamination/100)
    model.fit(data[["amount"]])
    return model, data

# -------------------------
# CALCULATE USER-DEFINED MEAN & STD
# -------------------------
mean_transaction = np.mean([sample1, sample2, sample3])
std_transaction = (std_percent/100) * mean_transaction

if "model" not in st.session_state:
    st.session_state.model, st.session_state.train_data = train_model(
        mean_transaction, std_transaction, sensitivity
    )

# -------------------------
# RETRAIN BUTTON
# -------------------------
if st.sidebar.button("Retrain Model"):
    st.session_state.model, st.session_state.train_data = train_model(
        mean_transaction, std_transaction, sensitivity
    )
    st.success("Model Retrained Successfully")

model = st.session_state.model

# -------------------------
# REAL-TIME TRANSACTION SIMULATION
# -------------------------
if st.button("Generate Live Transaction"):

    new_amount = abs(np.random.normal(mean_transaction, 2*std_transaction))
    
    new_tx = pd.DataFrame({
        "amount": [new_amount],
        "sender": [generate_account_id()],
        "receiver": [generate_account_id()]
    })

    score_raw = model.decision_function(new_tx[["amount"]])[0]
    risk_score = round((1 - score_raw) * 50, 2)

    # Classification logic
    if risk_score > 70:
        color = "red"
        label = "Suspicious"
    elif risk_score > 50:
        color = "yellow"
        label = "Potential"
    else:
        color = "green"
        label = "Normal"

    st.subheader("Current Transaction Analysis")
    st.write(f"Sender: {new_tx['sender'][0]}")
    st.write(f"Receiver: {new_tx['receiver'][0]}")
    st.write(f"Amount: ₹{round(new_amount,2)}")
    st.write(f"Risk Score: {risk_score}")
    st.markdown(f"### Status: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)

    # Store suspicious & potential only
    if label != "Normal":
        if "flagged" not in st.session_state:
            st.session_state.flagged = pd.DataFrame(columns=["Sender","Receiver","Amount","Risk","Category"])
        
        new_row = pd.DataFrame([{
            "Sender": new_tx['sender'][0],
            "Receiver": new_tx['receiver'][0],
            "Amount": round(new_amount,2),
            "Risk": risk_score,
            "Category": label
        }])
        
        st.session_state.flagged = pd.concat([st.session_state.flagged, new_row], ignore_index=True)

# -------------------------
# FLAGGED TABLE DISPLAY
# -------------------------
st.markdown("---")
st.subheader("Flagged Transactions (Suspicious & Potential Only)")

if "flagged" in st.session_state:
    st.dataframe(st.session_state.flagged)
else:
    st.write("No flagged transactions yet.")

# -------------------------
# PERFORMANCE SECTION
# -------------------------
st.markdown("---")
st.header("Our Performance")

if st.button("Show Performance"):

    test_data = pd.DataFrame({
        "amount": np.abs(np.random.normal(mean_transaction, std_transaction, 1000))
    })

    true_labels = (test_data["amount"] > mean_transaction + 2*std_transaction).astype(int)
    
    predictions = model.predict(test_data)
    pred_labels = np.where(predictions == -1, 1, 0)

    accuracy = accuracy_score(true_labels, pred_labels)
    coverage = sum(pred_labels)/len(pred_labels)

    fig = go.Figure(data=[go.Scatter3d(
        x=test_data["amount"][:100],
        y=[mean_transaction]*100,
        z=pred_labels[:100],
        mode='markers',
        marker=dict(
            size=5,
            color=pred_labels[:100],
            colorscale='Viridis'
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='Transaction Amount',
            yaxis_title='Mean Level',
            zaxis_title='Flagged (1=Yes)'
        ),
        height=600
    )

    st.plotly_chart(fig)
    st.write(f"Accuracy: {round(accuracy*100,2)}%")
    st.write(f"Coverage Ratio: {round(coverage*100,2)}% of transactions flagged")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Model Configuration")

sample1 = st.sidebar.number_input("Sample Transaction 1", min_value=0.0)
sample2 = st.sidebar.number_input("Sample Transaction 2", min_value=0.0)
sample3 = st.sidebar.number_input("Sample Transaction 3", min_value=0.0)

std_percentage = st.sidebar.slider("Std Deviation (% of Mean)", 1, 100, 20)
sensitivity = st.sidebar.slider("Suspicious Sensitivity (%)", 1, 20, 5)

# ---------------- HELPER FUNCTION ----------------
def generate_account_id():
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    numbers = ''.join(random.choices(string.digits, k=2))
    return letters + numbers

# ---------------- MODEL TRAINING ----------------
if sample1 and sample2 and sample3:

    mean_amount = np.mean([sample1, sample2, sample3])
    std_amount = (std_percentage/100) * mean_amount

    historical_data = pd.DataFrame({
        'amount': np.random.normal(mean_amount, std_amount, 1000)
    })

    historical_data['amount'] = historical_data['amount'].abs()

    model = IsolationForest(contamination=sensitivity/100)
    model.fit(historical_data[['amount']])

    st.success("Model Trained Successfully")

    # ---------------- LIVE SIMULATION ----------------
    if st.button("Generate Live Transaction"):

        new_std = 3 * std_amount

        new_tx_amount = abs(np.random.normal(mean_amount, new_std))

        sender = generate_account_id()
        receiver = generate_account_id()

        new_tx = pd.DataFrame({
            'amount':[new_tx_amount]
        })

        score = model.decision_function(new_tx)[0]
        risk_score = round((1 - score) * 50, 2)

        # Classification Logic
        if risk_score > 60:
            category = "Suspicious"
            color = "red"
        elif risk_score > 40:
            category = "Potential"
            color = "orange"
        else:
            category = "Normal"
            color = "green"

        # Display Current Transaction
        st.subheader("Current Transaction Analysis")
        col1, col2, col3 = st.columns(3)

        col1.metric("Amount", round(new_tx_amount,2))
        col2.metric("Risk Score", risk_score)
        col3.markdown(f"<h4 style='color:{color}'>{category}</h4>", unsafe_allow_html=True)

        # Store Suspicious & Potential Only
        if "alerts" not in st.session_state:
            st.session_state.alerts = pd.DataFrame(columns=["Sender","Receiver","Amount","Risk Score","Category"])

        if category in ["Suspicious","Potential"]:
            new_row = pd.DataFrame({
                "Sender":[sender],
                "Receiver":[receiver],
                "Amount":[round(new_tx_amount,2)],
                "Risk Score":[risk_score],
                "Category":[category]
            })
            st.session_state.alerts = pd.concat([st.session_state.alerts,new_row], ignore_index=True)

        # Alert Table
        st.subheader("Flagged Transactions")
        st.dataframe(st.session_state.alerts)

    # ---------------- PERFORMANCE SECTION ----------------
    st.subheader("Model Performance Evaluation")

    if st.button("Show Performance"):

        test_data = pd.DataFrame({
            'amount': np.random.normal(mean_amount, std_amount, 1000)
        })

        test_data['amount'] = test_data['amount'].abs()

        # Artificially inject anomalies
        anomaly_indices = np.random.choice(1000, 50, replace=False)
        test_data.loc[anomaly_indices,'amount'] *= 4

        y_true = np.zeros(1000)
        y_true[anomaly_indices] = 1

        predictions = model.predict(test_data[['amount']])
        y_pred = np.where(predictions==-1,1,0)

        f1 = round(f1_score(y_true, y_pred),2)

        # 3D Visualization
        test_data['Risk'] = model.decision_function(test_data[['amount']])
        test_data['Index'] = range(1000)

        fig = px.scatter_3d(
            test_data,
            x="Index",
            y="amount",
            z="Risk",
            color=y_pred.astype(str),
            title="3D Risk Distribution Map"
        )

        st.plotly_chart(fig)

        st.success(f"Model F1 Score on Test Data: {f1}")
