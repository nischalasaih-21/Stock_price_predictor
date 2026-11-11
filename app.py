
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load model
model = joblib.load(BASE_DIR / "churn_model.pkl")

st.set_page_config(page_title="Customer Churn Predictor",
                   page_icon="📊")

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer will leave a service.")

st.subheader("Enter Customer Details")

tenure = st.slider("Customer Tenure (Months)", 0, 72, 12)

charges = st.number_input(
    "Monthly Charges",
    min_value=10.0,
    max_value=200.0,
    value=80.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

support = st.selectbox(
    "Technical Support",
    ["Yes", "No"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

if st.button("Predict Churn"):

    customer = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": charges,
        "Contract": contract,
        "InternetService": internet,
        "TechSupport": support,
        "PaperlessBilling": paperless
    }])

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Customer is likely to leave!")
    else:
        st.success("Customer is likely to stay!")

    st.write(
        "Churn Probability:",
        round(probability * 100, 2), "%"
    )
