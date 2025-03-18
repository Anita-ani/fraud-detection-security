import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("fraud_model.pkl")  # Ensure you have a trained model saved as 'fraud_model.pkl'

st.title("Fraud Detection System")
st.write("This app predicts whether a transaction is fraudulent or not.")

# File upload section
uploaded_file = st.file_uploader("Upload a CSV file with transactions", type=["csv"])

if uploaded_file is not None:
    # Load and preview data
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(data.head())

    # Ensure the data has all the required features
    required_features = model.feature_names_in_
    
    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Apply Label Encoding to categorical columns
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

    # Add missing features (if any) with default values
    for col in required_features:
        if col not in data.columns:
            data[col] = 0  # Fill missing features

    # Ensure correct column order
    data = data[required_features]

    # Fraud prediction button
    if st.button("Predict Fraudulent Transactions"):
        predictions = model.predict(data)
        data["Fraud Prediction"] = predictions
        st.write("Fraud Detection Results:")
        st.dataframe(data)

        # Provide download option
        st.download_button("Download Predictions", data.to_csv(index=False), "fraud_prediction
