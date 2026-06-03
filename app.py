import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("--- Step 1: Loading Dataset ---")
# Because main.py and the CSV are in the same folder, we can just use the filename
try:
    df = pd.read_csv("Loan_default.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(
        "ERROR: CSV file not found. Make sure this script and Loan_default.csv are in the exact same folder."
    )
    exit()

# Drop unique identifier columns that don't help classification
X = df.drop(columns=["LoanID", "Loan_default"], errors="ignore")
y = df["Loan_default"]

# Convert any text columns (like Education or Marital Status) into numbers automatically
X = pd.get_dummies(X, drop_first=True)

print("\n--- Step 2: Splitting Data ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- Step 3: Scaling Numerical Features ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Step 4: Applying SMOTE (Balancing Classes) ---")
print(f"Original class counts: {np.bincount(y_train)}")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"Balanced class counts: {np.bincount(y_train_res)}")

print("\n--- Step 5: Training Random Forest Classifier ---")
# n_jobs=-1 forces the model to use all CPU cores of your smartphone
model = RandomForestClassifier(
    n_estimators=50, max_depth=10, n_jobs=-1, random_state=42
)
model.fit(X_train_res, y_train_res)

print("\n--- Step 6: Generating Evaluation Metrics ---")
y_pred = model.predict(X_test_scaled)

print("\n[Confusion Matrix]")
print(confusion_matrix(y_test, y_pred))

print("\n[Classification Report]")
print(classification_report(y_test, y_pred))
