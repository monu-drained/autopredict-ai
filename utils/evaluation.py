import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

def evaluate_classification(model, X_test, y_test, n_classes):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc, fpr, tpr = None, None, None
    try:
        if n_classes == 2:
            proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
            fpr, tpr, _ = roc_curve(y_test, proba)
        else:
            proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, proba, multi_class="ovr")
    except Exception:
        pass
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return {"pred": y_pred, "acc": acc, "auc": auc, "fpr": fpr, "tpr": tpr, "report": report, "cm": cm}

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"pred": y_pred, "mae": mae, "rmse": rmse, "r2": r2}

def build_leaderboard_classification(results):
    rows = []
    for name, r in results.items():
        rep = r["report"]
        rows.append({
            "Model": name,
            "Accuracy": round(r["acc"], 4),
            "AUC": round(r["auc"], 4) if r["auc"] else "N/A",
            "Precision": round(rep["macro avg"]["precision"], 4),
            "Recall": round(rep["macro avg"]["recall"], 4),
            "F1": round(rep["macro avg"]["f1-score"], 4),
        })
    df = pd.DataFrame(rows)
    if "Accuracy" in df.columns:
        df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return df

def build_leaderboard_regression(results):
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "MAE": round(r["mae"], 4),
            "RMSE": round(r["rmse"], 4),
            "R² Score": round(r["r2"], 4),
        })
    df = pd.DataFrame(rows)
    if "R² Score" in df.columns:
        df = df.sort_values("R² Score", ascending=False).reset_index(drop=True)
    return df
