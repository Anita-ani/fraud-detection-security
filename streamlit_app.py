import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("fraud_model.pkl")  # Ensure you have a trained model saved as 'fraud_model.pkl'

st.title("Fraud Detection System")
st.write("This app predicts whether a transaction is fraudulent or not.")

# User input section
uploaded_file = st.file_uploader("Upload a CSV file with transactions", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(data.head())

    if st.button("Predict Fraudulent Transactions"):
        predictions = model.predict(data)
        data["Fraud Prediction"] = predictions
        st.write("Fraud Detection Results:")
        st.dataframe(data)
        st.download_button("Download Predictions", data.to_csv(index=False), "fraud_predictions.csv", "text/csv")
