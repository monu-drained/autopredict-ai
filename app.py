import io
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from utils.preprocessing import detect_task_type, detect_target_column, clean_dataframe, prepare_features, apply_feature_engineering, profile_dataframe
from utils.training import get_model_options, build_model, tune_model
from utils.evaluation import evaluate_classification, evaluate_regression, build_leaderboard_classification, build_leaderboard_regression
from utils.visualization import plot_roc_curves, plot_confusion_matrix, plot_accuracy_bar, plot_regression_metrics, plot_pred_vs_actual, plot_residuals, plot_feature_importance, plot_distribution, plot_correlation_heatmap

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="AutoPredict AI v2", page_icon="🔮", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');
:root {--bg:#0a0a0f;--surface:#12121a;--surface2:#1a1a26;--accent:#7c3aed;--accent2:#06b6d4;--text:#e2e8f0;--muted:#64748b;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg)!important;color:var(--text)!important;}
.stApp{background-color:var(--bg)!important;}
header[data-testid="stHeader"]{background:transparent;}
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid #1e1e2e;}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
.stButton>button{background:linear-gradient(135deg,#7c3aed,#9333ea)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:700!important;padding:0.6rem 1.5rem!important;}
[data-testid="metric-container"]{background:var(--surface2)!important;border:1px solid #2d2d3d!important;border-radius:12px!important;padding:1rem!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:10px!important;padding:4px!important;}
.stTabs [aria-selected="true"]{background:#7c3aed!important;color:white!important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#12002f,#090016);padding:2rem;border-radius:24px;border:1px solid rgba(138,43,226,0.25);margin-bottom:1.5rem;">
<p style="color:#8b5cf6;letter-spacing:6px;font-size:0.8rem;margin-bottom:0.5rem;">• MACHINE LEARNING STUDIO v2</p>
<h1 style="font-family:'Space Mono',monospace;font-size:2.4rem;font-weight:700;margin:0.4rem 0 0.6rem;background:linear-gradient(90deg,#a855f7,#3b82f6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">AutoPredict AI 🔮</h1>
<p style="color:#94a3b8;">Upload any CSV → auto-detect task → engineer features → train & tune → explain with SHAP → export.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.divider()
    st.markdown("### ⚙️ Settings")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random seed", value=42, step=1)
    st.divider()
    st.markdown("### 🧬 Feature Engineering")
    fe_log = st.checkbox("Log-transform numeric features")
    fe_interact = st.checkbox("Add pairwise interactions")
    fe_poly = st.checkbox("Polynomial features (degree 2)")
    st.divider()
    st.markdown("### 🔧 Hyperparameter Tuning")
    do_tuning = st.checkbox("Enable tuning (RandomizedSearchCV)")
    tuning_iters = st.slider("Search iterations", 5, 50, 10, 5) if do_tuning else 10
    tuning_cv = st.slider("CV folds", 2, 5, 3) if do_tuning else 3
    st.divider()
    st.markdown("### 💡 SHAP Explainability")
    do_shap = st.checkbox("Generate SHAP explanations", value=False)
    if do_shap and not SHAP_AVAILABLE:
        st.warning("SHAP not installed. Run: pip install shap")
        do_shap = False

if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

df, dropped_cols = clean_dataframe(df_raw)
if dropped_cols:
    st.warning(f"Auto-dropped columns: {', '.join(dropped_cols)}")

default_target = detect_target_column(df)
col1, col2, col3 = st.columns([3,1,1])
with col1:
    target_col = st.selectbox("🎯 Target column", options=df.columns.tolist(), index=df.columns.tolist().index(default_target))
with col2:
    auto_task = detect_task_type(df[target_col])
    task_type = st.selectbox("Task type", ["classification","regression"], index=0 if auto_task=="classification" else 1)
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶ Run Analysis", use_container_width=True)

model_options = get_model_options(task_type)
with st.sidebar:
    st.divider()
    st.markdown("### 🤖 Model Selection")
    models_selected = st.multiselect("Choose models", model_options, default=model_options[:2])

with st.expander("📊 Dataset Profiling", expanded=False):
    prof_tab1, prof_tab2, prof_tab3 = st.tabs(["Summary","Distributions","Correlation"])
    with prof_tab1:
        st.markdown(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
        st.dataframe(profile_dataframe(df_raw), use_container_width=True)
    with prof_tab2:
        col_to_plot = st.selectbox("Select column", df_raw.columns.tolist(), key="dist_col")
        fig_dist = plot_distribution(df_raw[col_to_plot], col_to_plot)
        st.pyplot(fig_dist)
        plt.close()
    with prof_tab3:
        fig_corr = plot_correlation_heatmap(df_raw)
        if fig_corr:
            st.pyplot(fig_corr)
            plt.close()
        else:
            st.info("Not enough numeric columns for a heatmap.")

if not (run_btn or st.session_state.get("ran_once")):
    st.stop()

st.session_state["ran_once"] = True

if not models_selected:
    st.error("Select at least one model from the sidebar.")
    st.stop()

n_classes = df[target_col].nunique()
if task_type == "classification" and n_classes > 20:
    st.error(f"Too many target classes ({n_classes}) for classification.")
    st.stop()

try:
    X_raw, y, target_classes = prepare_features(df, target_col, task_type)
except Exception as e:
    st.error(f"Feature preparation failed: {e}")
    st.stop()

X = apply_feature_engineering(X_raw, add_log=fe_log, add_interactions=fe_interact, add_poly=fe_poly)

stratify = y if (task_type == "classification" and n_classes <= 10) else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=stratify)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}
best_params = {}
progress = st.progress(0, text="Training models…")

for i, name in enumerate(models_selected):
    model = build_model(name, task_type)
    if model is None:
        st.warning(f"Could not build model: {name}")
        continue
    if do_tuning:
        with st.spinner(f"Tuning {name}…"):
            try:
                model, bp = tune_model(model, name, X_train_s, y_train, task_type, n_iter=tuning_iters, cv=tuning_cv)
                best_params[name] = bp
            except Exception as ex:
                st.warning(f"Tuning failed for {name}: {ex}. Using default params.")
                model.fit(X_train_s, y_train)
    else:
        model.fit(X_train_s, y_train)
    if task_type == "classification":
        res = evaluate_classification(model, X_test_s, y_test, n_classes)
    else:
        res = evaluate_regression(model, X_test_s, y_test)
    res["model"] = model
    results[name] = res
    progress.progress((i+1)/len(models_selected), text=f"{name} ✓")

progress.empty()
st.success(f"✅ Trained {len(results)} model(s)")

if task_type == "classification":
    best_name = max(results, key=lambda n: results[n]["acc"])
    best_metric_label = "Accuracy"
    best_metric_val = f"{results[best_name]['acc']:.2%}"
else:
    best_name = max(results, key=lambda n: results[n]["r2"])
    best_metric_label = "R²"
    best_metric_val = f"{results[best_name]['r2']:.4f}"

st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a0a2e,#0a1628);border:1px solid #7c3aed;border-radius:12px;padding:1.5rem;margin:1rem 0;text-align:center;">
<div style="font-family:'Space Mono',monospace;font-size:1.6rem;color:#e2e8f0;font-weight:700;">🏆 Best Model: {best_name}</div>
<div style="color:#06b6d4;font-size:1rem;margin-top:0.4rem;">{best_metric_label}: <b>{best_metric_val}</b></div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🏅 Leaderboard","📊 Charts","🔢 Metrics","🔬 Feature Importance","🧠 SHAP","📦 Export"])
t_leader, t_charts, t_metrics, t_fi, t_shap, t_export = tabs

with t_leader:
    st.markdown("### 🏅 Model Leaderboard")
    if task_type == "classification":
        lb = build_leaderboard_classification(results)
    else:
        lb = build_leaderboard_regression(results)
    st.dataframe(lb, use_container_width=True)
    if best_params:
        with st.expander("🔧 Best Hyperparameters"):
            for name, bp in best_params.items():
                st.markdown(f"**{name}**")
                st.json(bp)

with t_charts:
    if task_type == "classification":
        if n_classes == 2:
            st.markdown("### ROC Curves")
            fig = plot_roc_curves(results)
            st.pyplot(fig); plt.close()
        st.markdown("### Accuracy Comparison")
        fig = plot_accuracy_bar(results)
        st.pyplot(fig); plt.close()
        st.markdown("### Confusion Matrices")
        cols = st.columns(min(len(results), 3))
        for idx, (name, r) in enumerate(results.items()):
            with cols[idx % len(cols)]:
                fig = plot_confusion_matrix(r["cm"], name)
                st.pyplot(fig); plt.close()
    else:
        st.markdown("### R² Score Comparison")
        fig = plot_regression_metrics(results)
        st.pyplot(fig); plt.close()
        st.markdown("### Predicted vs Actual")
        cols = st.columns(min(len(results), 2))
        for idx, (name, r) in enumerate(results.items()):
            with cols[idx % len(cols)]:
                fig = plot_pred_vs_actual(y_test, r["pred"], name)
                st.pyplot(fig); plt.close()
        st.markdown("### Residual Plots")
        cols2 = st.columns(min(len(results), 2))
        for idx, (name, r) in enumerate(results.items()):
            with cols2[idx % len(cols2)]:
                fig = plot_residuals(y_test, r["pred"], name)
                st.pyplot(fig); plt.close()

with t_metrics:
    for name, r in results.items():
        st.markdown(f"### {name}")
        if task_type == "classification":
            c1, c2 = st.columns(2)
            with c1:
                rep_df = pd.DataFrame(r["report"]).transpose().round(3)
                st.dataframe(rep_df, use_container_width=True)
            with c2:
                cm_df = pd.DataFrame(r["cm"])
                st.dataframe(cm_df, use_container_width=True)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{r['mae']:.4f}")
            c2.metric("RMSE", f"{r['rmse']:.4f}")
            c3.metric("R²", f"{r['r2']:.4f}")
        st.divider()

with t_fi:
    st.markdown("### 🔬 Feature Importance")
    top_n = st.slider("Top N features", 5, min(30, len(X.columns)), 15, key="fi_n")
    for name, r in results.items():
        fig = plot_feature_importance(r["model"], X.columns.tolist(), name, top_n)
        if fig:
            st.pyplot(fig); plt.close()
        else:
            st.info(f"{name} does not expose feature importances.")

with t_shap:
    st.markdown("### 🧠 SHAP Explainability")
    if not do_shap:
        st.info("Enable SHAP in the sidebar to generate explanations.")
    elif not SHAP_AVAILABLE:
        st.warning("Install SHAP: pip install shap")
    else:
        shap_model_name = st.selectbox("Model to explain", list(results.keys()), key="shap_model")
        shap_model = results[shap_model_name]["model"]
        max_shap_samples = min(200, X_test_s.shape[0])
        with st.spinner("Computing SHAP values…"):
            try:
                X_shap = pd.DataFrame(X_test_s[:max_shap_samples], columns=X.columns)
                if hasattr(shap_model, "feature_importances_"):
                    explainer = shap.TreeExplainer(shap_model)
                    shap_values = explainer.shap_values(X_shap)
                    sv = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values
                else:
                    explainer = shap.LinearExplainer(shap_model, X_shap)
                    sv = explainer.shap_values(X_shap)
                fig_s, ax_s = plt.subplots(figsize=(8,5))
                fig_s.patch.set_facecolor("#12121a")
                shap.summary_plot(sv, X_shap, plot_type="bar", show=False, color="#7c3aed")
                plt.title(f"SHAP Feature Importance — {shap_model_name}", color="#e2e8f0")
                st.pyplot(fig_s); plt.close()
                fig_b, ax_b = plt.subplots(figsize=(8,5))
                fig_b.patch.set_facecolor("#12121a")
                shap.summary_plot(sv, X_shap, show=False)
                plt.title(f"SHAP Beeswarm — {shap_model_name}", color="#e2e8f0")
                st.pyplot(fig_b); plt.close()
            except Exception as ex:
                st.error(f"SHAP failed: {ex}")

with t_export:
    st.markdown("### 📦 Export & Deployment")
    st.markdown("#### 📥 Download Leaderboard (CSV)")
    if task_type == "classification":
        lb_dl = build_leaderboard_classification(results)
    else:
        lb_dl = build_leaderboard_regression(results)
    st.download_button("⬇ Download Leaderboard CSV", data=lb_dl.to_csv(index=False), file_name="leaderboard.csv", mime="text/csv")
    st.markdown("#### 📥 Download Predictions (CSV)")
    pred_df = pd.DataFrame({"y_true": y_test.values})
    for name, r in results.items():
        pred_df[f"pred_{name}"] = r["pred"]
    st.download_button("⬇ Download Predictions CSV", data=pred_df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
    st.markdown("#### 💾 Download Best Model (Pickle)")
    best_model_obj = results[best_name]["model"]
    model_bytes = pickle.dumps({"model": best_model_obj, "scaler": scaler, "feature_names": X.columns.tolist()})
    st.download_button(f"⬇ Download {best_name} (.pkl)", data=model_bytes, file_name=f"{best_name.replace(' ','_').lower()}_model.pkl", mime="application/octet-stream")

st.markdown("""
<div style="text-align:center;color:#2d2d3d;font-size:0.75rem;margin-top:3rem;padding-top:1rem;border-top:1px solid #1e1e2e;">
AutoPredict AI v2 · Built with Streamlit · Powered by scikit-learn, XGBoost, LightGBM, SHAP
</div>
""", unsafe_allow_html=True)
