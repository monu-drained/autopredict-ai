import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def get_model_options(task_type):
    base_clf = ["Random Forest", "Gradient Boosting", "Logistic Regression"]
    base_reg = ["Random Forest Regressor", "Gradient Boosting Regressor", "Linear Regression", "Ridge Regression"]
    if XGBOOST_AVAILABLE:
        base_clf.append("XGBoost")
        base_reg.append("XGBoost Regressor")
    if LIGHTGBM_AVAILABLE:
        base_clf.append("LightGBM")
        base_reg.append("LightGBM Regressor")
    return base_clf if task_type == "classification" else base_reg

def build_model(name, task_type):
    clf_map = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    }
    reg_map = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
    }
    if XGBOOST_AVAILABLE:
        clf_map["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
        reg_map["XGBoost Regressor"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    if LIGHTGBM_AVAILABLE:
        clf_map["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        reg_map["LightGBM Regressor"] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model_map = clf_map if task_type == "classification" else reg_map
    return model_map.get(name)

PARAM_GRIDS = {
    "Random Forest": {"n_estimators": [50,100,200], "max_depth": [6,10,15,None], "min_samples_split": [2,5,10], "max_features": ["sqrt","log2"]},
    "Random Forest Regressor": {"n_estimators": [50,100,200], "max_depth": [6,10,15,None], "min_samples_split": [2,5,10]},
    "Gradient Boosting": {"n_estimators": [50,100,200], "learning_rate": [0.01,0.05,0.1,0.2], "max_depth": [3,5,7], "subsample": [0.7,0.85,1.0]},
    "Gradient Boosting Regressor": {"n_estimators": [50,100,200], "learning_rate": [0.01,0.05,0.1,0.2], "max_depth": [3,5,7]},
    "Logistic Regression": {"C": [0.01,0.1,1,10,100], "solver": ["lbfgs","saga"], "penalty": ["l2"]},
    "Ridge Regression": {"alpha": [0.01,0.1,1.0,10.0,100.0]},
    "XGBoost": {"n_estimators": [50,100,200], "learning_rate": [0.01,0.05,0.1,0.2], "max_depth": [3,5,7], "subsample": [0.7,0.85,1.0]},
    "XGBoost Regressor": {"n_estimators": [50,100,200], "learning_rate": [0.01,0.05,0.1], "max_depth": [3,5,7]},
    "LightGBM": {"n_estimators": [50,100,200], "learning_rate": [0.01,0.05,0.1], "num_leaves": [15,31,63]},
    "LightGBM Regressor": {"n_estimators": [50,100,200], "learning_rate": [0.01,0.05,0.1], "num_leaves": [15,31,63]},
}

def tune_model(model, name, X_train, y_train, task_type, n_iter=20, cv=3):
    param_grid = PARAM_GRIDS.get(name)
    if not param_grid:
        return model, {}
    scoring = "f1_weighted" if task_type == "classification" else "neg_root_mean_squared_error"
    cv_split = StratifiedKFold(n_splits=cv) if task_type == "classification" else KFold(n_splits=cv)
    search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, scoring=scoring, cv=cv_split, random_state=42, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
