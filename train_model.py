
# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 2. Load dataset
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "customer_churn.csv"

df = pd.read_csv(data_path)

print("Dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# 3. Remove unnecessary customer ID
df = df.drop("customerID", axis=1)

# 4. Separate input and output
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# 5. Identify numerical and categorical columns
numerical_features = ["tenure", "MonthlyCharges"]

categorical_features = [
    "Contract",
    "InternetService",
    "TechSupport",
    "PaperlessBilling"
]

# 6. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)

# 7. Create ML pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 8. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Train model
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# 10. Predict test data
y_pred = model.predict(X_test)

# 11. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 12. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Customer Churn Confusion Matrix")
plt.tight_layout()
plt.savefig(BASE_DIR / "confusion_matrix.png")
plt.show()

# 13. Save trained model
model_path = BASE_DIR / "churn_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully at:", model_path)