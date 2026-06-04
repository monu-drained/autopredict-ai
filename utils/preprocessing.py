import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

def detect_task_type(series, n_unique_threshold=15):
    if series.dtype == object:
        return "classification"
    if series.nunique() <= n_unique_threshold:
        return "classification"
    return "regression"

def detect_target_column(df):
    keywords = ["default","loan_default","status","label","target","outcome","churn","fraud","result","class","approved","survived","y","output","prediction","price","value","salary","revenue","sales","score"]
    for col in df.columns:
        if col.lower() in keywords:
            return col
    return df.columns[-1]

def clean_dataframe(df):
    df = df.copy()
    cols_to_drop = []
    id_patterns = ["id","uuid","customerid","customer_id","userid","user_id","loan_id"]
    for col in df.columns:
        if col.lower() in id_patterns:
            cols_to_drop.append(col)
            continue
        if df[col].dtype == object:
            n_unique = df[col].nunique()
            if n_unique > 50 or n_unique == len(df):
                cols_to_drop.append(col)
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    return df, cols_to_drop

def apply_feature_engineering(X, add_log=False, add_interactions=False, add_poly=False, poly_degree=2):
    X = X.copy()
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    if add_log:
        for col in numeric_cols:
            if (X[col] > 0).all():
                X[f"log_{col}"] = np.log1p(X[col])
    if add_interactions and len(numeric_cols) >= 2:
        top_cols = numeric_cols[:5]
        for i, c1 in enumerate(top_cols):
            for c2 in top_cols[i+1:]:
                X[f"{c1}_x_{c2}"] = X[c1] * X[c2]
    if add_poly and len(numeric_cols) >= 1:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        top_cols = numeric_cols[:5]
        poly_arr = poly.fit_transform(X[top_cols])
        poly_names = poly.get_feature_names_out(top_cols)
        poly_df = pd.DataFrame(poly_arr, columns=poly_names, index=X.index)
        X = X.drop(columns=top_cols)
        X = pd.concat([X, poly_df], axis=1)
    return X

def prepare_features(df, target_col, task_type):
    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col].copy()
    target_classes = None
    if task_type == "classification":
        if y.dtype == object:
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=target_col)
            target_classes = list(le.classes_)
        else:
            y = y.astype(int)
            target_classes = sorted(y.unique().tolist())
    else:
        y = y.astype(float)
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X, y, target_classes

def profile_dataframe(df):
    rows = []
    for col in df.columns:
        info = {"Column": col, "DType": str(df[col].dtype), "Missing %": round(df[col].isna().mean()*100, 2), "Unique": df[col].nunique()}
        if pd.api.types.is_numeric_dtype(df[col]):
            info["Mean"] = round(df[col].mean(), 4)
            info["Std"] = round(df[col].std(), 4)
            info["Min"] = round(df[col].min(), 4)
            info["Max"] = round(df[col].max(), 4)
        else:
            info["Mean"] = info["Std"] = info["Min"] = info["Max"] = "—"
        rows.append(info)
    return pd.DataFrame(rows)
