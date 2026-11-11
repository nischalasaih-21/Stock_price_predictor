
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load saved model
model = joblib.load(BASE_DIR / "churn_model.pkl")

# New customer details
new_customer = pd.DataFrame([{
    "tenure": 3,
    "MonthlyCharges": 95,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "TechSupport": "No",
    "PaperlessBilling": "Yes"
}])

# Make prediction
prediction = model.predict(new_customer)[0]

probability = model.predict_proba(new_customer)[0][1]

print("\nCustomer Churn Prediction")

if prediction == 1:
    print("Result: Customer is likely to leave.")
else:
    print("Result: Customer is likely to stay.")

print("Churn Probability:", round(probability * 100, 2), "%")
