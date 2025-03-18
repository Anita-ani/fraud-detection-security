import streamlit as st
import joblib
import pandas as pd

# Load trained fraud detection model
model = joblib.load("fraud_detection_model.pkl")

# Streamlit UI
st.title("🔍 Fraud Detection System")
st.write("Enter transaction details to predict fraud.")

# Input fields
amount = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f")
old_balance = st.number_input("Account Balance Before Transaction ($)", min_value=0.0, format="%.2f")
new_balance = st.number_input("Account Balance After Transaction ($)", min_value=0.0, format="%.2f")

# Prediction logic
if st.button("Predict Fraud"):
    input_data = pd.DataFrame([[amount, old_balance, new_balance]],
                              columns=["Amount", "OldBalanceOrig", "NewBalanceOrig"])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")

