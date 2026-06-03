import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("🏦 Loan Default Prediction")
st.write("Upload your dataset to train and evaluate the model.")

uploaded_file = st.file_uploader("Upload Loan_default.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded!")
    st.write("**Preview:**", df.head())

    # --- Target column detection ---
    possible_targets = ["Loan_default","Default","default","LoanStatus","Status","loan_default"]
    target_col = next((c for c in possible_targets if c in df.columns), df.columns[-1])
    st.info(f"Target column detected: **{target_col}**")

    # --- Features ---
    id_cols = [col for col in df.columns if "ID" in col or "id" in col]
    X = df.drop(columns=id_cols + [target_col], errors="ignore")
    y = df[target_col]
    X = pd.get_dummies(X, drop_first=True)

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Scale ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    st.write("**Class counts (training set):**", dict(zip(*np.unique(y_train, return_counts=True))))

    # --- Train ---
    with st.spinner("Training Random Forest... please wait ⏳"):
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"
        )
        model.fit(X_train_scaled, y_train)
    st.success("✅ Model trained!")

    # --- Evaluate ---
    y_pred = model.predict(X_test_scaled)

    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm))

    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))
