import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoPredict AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --accent: #7c3aed;
    --accent2: #06b6d4;
    --accent3: #f59e0b;
    --text: #e2e8f0;
    --muted: #64748b;
    --success: #10b981;
    --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg) !important; }

/* Hide default streamlit header */
header[data-testid="stHeader"] { background: transparent; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid #1e1e2e;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Inputs & selectboxes */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--surface2) !important;
    border: 1px solid #2d2d3d !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

.stFileUploader {
    background: var(--surface2) !important;
    border: 2px dashed var(--accent) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #9333ea) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.5rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid #2d2d3d !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent2) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem !important;
}

/* DataFrames */
.stDataFrame {
    border: 1px solid #2d2d3d !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Alerts */
.stSuccess { background: rgba(16,185,129,0.1) !important; border: 1px solid var(--success) !important; border-radius: 8px !important; }
.stError   { background: rgba(239,68,68,0.1)  !important; border: 1px solid var(--danger)  !important; border-radius: 8px !important; }
.stInfo    { background: rgba(6,182,212,0.1)   !important; border: 1px solid var(--accent2) !important; border-radius: 8px !important; }
.stWarning { background: rgba(245,158,11,0.1)  !important; border: 1px solid var(--accent3) !important; border-radius: 8px !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #12121a 0%, #1a0a2e 50%, #0a1628 100%);
    border: 1px solid #2d2d3d;
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
">
    <div style="
        position: absolute; top: -40px; right: -40px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(124,58,237,0.15), transparent 70%);
        border-radius: 50%;
    "></div>
    <div style="
        position: absolute; bottom: -30px; left: 20%;
        width: 150px; height: 150px;
        background: radial-gradient(circle, rgba(6,182,212,0.1), transparent 70%);
        border-radius: 50%;
    "></div>
    <span style="
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #7c3aed;
        letter-spacing: 0.2em;
        text-transform: uppercase;
    ">● MACHINE LEARNING STUDIO</span>
    <h1 style="
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0.4rem 0 0.6rem;
        background: linear-gradient(135deg, #e2e8f0, #7c3aed, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    ">AutoPredict AI 🔮</h1>
    <p style="color: #94a3b8; margin: 0; font-size: 1rem; max-width: 500px;">
        Upload any CSV → auto-detect target → train multiple models → get predictions. No code needed.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def detect_target_column(df):
    """Try to auto-detect the most likely target/label column."""
    keywords = [
        "default", "loan_default", "status", "label", "target",
        "outcome", "churn", "fraud", "result", "class", "approved",
        "survived", "y", "output", "prediction"
    ]
    for col in df.columns:
        if col.lower() in keywords:
            return col
    # Fall back: last column
    return df.columns[-1]

def clean_dataframe(df):
    """Drop high-cardinality text columns, encode categoricals."""
    df = df.copy()
    cols_to_drop = []
    for col in df.columns:
        if df[col].dtype == object:
            n_unique = df[col].nunique()
            if n_unique > 50 or n_unique == len(df):
                cols_to_drop.append(col)
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    return df, cols_to_drop

def prepare_features(df, target_col):
    """Encode and split features/target."""
    id_cols = [c for c in df.columns if c.lower() in ["id","loan_id","customerid","customer_id","userid","user_id"]]
    drop_cols = id_cols + [target_col]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col].copy()

    # Encode target if needed
    if y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_col)
        target_classes = le.classes_
    else:
        y = y.astype(int)
        target_classes = sorted(y.unique())

    # One-hot encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    return X, y, target_classes

def make_dark_fig():
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#12121a")
    ax.set_facecolor("#1a1a26")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines["bottom"].set_color("#2d2d3d")
    ax.spines["left"].set_color("#2d2d3d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.65rem;
    color:#7c3aed; letter-spacing:0.15em; text-transform:uppercase;
    margin-bottom:1rem;">⚙ Configuration</div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Any CSV file",
        type=["csv"],
        help="Upload any CSV — the app will figure out the rest."
    )

    st.divider()
    st.markdown("### 🎯 Model Settings")

    models_selected = st.multiselect(
        "Choose models to train",
        ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        default=["Random Forest", "Gradient Boosting"]
    )

    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05,
                          help="Fraction of data used for testing")

    st.divider()
    st.markdown("""
    <div style="font-size:0.75rem; color:#64748b; line-height:1.6;">
    <b style="color:#94a3b8;">How it works:</b><br>
    1. Upload any CSV<br>
    2. Auto-detects target column<br>
    3. Cleans & encodes data<br>
    4. Trains selected models<br>
    5. Shows metrics & charts
    </div>
    """, unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────

if uploaded_file is None:
    # Landing state
    col1, col2, col3 = st.columns(3)
    cards = [
        ("🧹", "Auto Clean", "Drops ID cols, encodes text, fills missing values automatically"),
        ("🤖", "Multi-Model", "Train Random Forest, Gradient Boosting & Logistic Regression side by side"),
        ("📊", "Rich Metrics", "Accuracy, AUC-ROC, confusion matrix, feature importance & more"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div style="background:#12121a; border:1px solid #2d2d3d; border-radius:12px;
            padding:1.5rem; text-align:center; height:160px;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-family:'Space Mono',monospace; font-weight:700;
                color:#e2e8f0; margin:0.5rem 0 0.4rem; font-size:0.9rem;">{title}</div>
                <div style="color:#64748b; font-size:0.8rem; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 **Upload a CSV file** from the sidebar to get started.")

else:
    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    # ── Clean ─────────────────────────────────────────────────────────────────
    df, dropped_cols = clean_dataframe(df_raw)
    if dropped_cols:
        st.warning(f"🗑️ Auto-dropped high-cardinality columns: `{'`, `'.join(dropped_cols)}`")

    # ── Detect target ─────────────────────────────────────────────────────────
    target_col = detect_target_column(df)

    # ── Manual override ───────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 1])
    with col_left:
        target_col = st.selectbox(
            "🎯 Target column (auto-detected — change if wrong)",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(target_col)
        )
    with col_right:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶ Run Analysis", width="stretch")

    # ── Data preview ─────────────────────────────────────────────────────────
    with st.expander("🔍 Dataset Preview", expanded=False):
        st.markdown(f"**Shape:** {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
        st.dataframe(df_raw.head(10), width="stretch")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df_raw.shape[0]:,}")
        c2.metric("Columns", df_raw.shape[1])
        c3.metric("Missing %", f"{(df_raw.isnull().sum().sum() / df_raw.size * 100):.1f}%")

    # ── Run on button press ───────────────────────────────────────────────────
    if run_btn or st.session_state.get("ran_once"):
        st.session_state["ran_once"] = True

        if not models_selected:
            st.error("Please select at least one model in the sidebar.")
            st.stop()

        # Check target is binary/multiclass with limited classes
        n_classes = df[target_col].nunique()
        if n_classes > 20:
            st.error(f"Target column `{target_col}` has {n_classes} unique values. "
                     "Please select a classification target (with few unique values).")
            st.stop()

        # Prepare
        with st.spinner("Preparing data..."):
            try:
                X, y, target_classes = prepare_features(df, target_col)
            except Exception as e:
                st.error(f"Data preparation failed: {e}")
                st.stop()

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if n_classes <= 10 else None
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # ── Train models ──────────────────────────────────────────────────────
        model_map = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=12, n_jobs=-1,
                random_state=42, class_weight="balanced"
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=4, random_state=42
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42, class_weight="balanced"
            ),
        }

        results = {}
        progress = st.progress(0, text="Training models...")

        for i, name in enumerate(models_selected):
            with st.spinner(f"Training {name}..."):
                m = model_map[name]
                m.fit(X_train_s, y_train)
                y_pred = m.predict(X_test_s)
                acc = accuracy_score(y_test, y_pred)

                try:
                    if n_classes == 2:
                        proba = m.predict_proba(X_test_s)[:, 1]
                        auc = roc_auc_score(y_test, proba)
                        fpr, tpr, _ = roc_curve(y_test, proba)
                    else:
                        proba = m.predict_proba(X_test_s)
                        auc = roc_auc_score(y_test, proba, multi_class="ovr")
                        fpr, tpr = None, None
                except Exception:
                    auc, fpr, tpr = None, None, None

                results[name] = {
                    "model": m, "pred": y_pred,
                    "acc": acc, "auc": auc,
                    "fpr": fpr, "tpr": tpr,
                    "report": classification_report(y_test, y_pred, output_dict=True),
                    "cm": confusion_matrix(y_test, y_pred)
                }
            progress.progress((i + 1) / len(models_selected),
                              text=f"✅ {name} done")

        progress.empty()
        st.success(f"🎉 Trained {len(models_selected)} model(s) successfully!")

        # ── Results tabs ──────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 Overview", "🔢 Detailed Metrics", "📊 Charts", "🔬 Feature Importance"]
        )

        # TAB 1 — Overview
        with tab1:
            st.markdown("### Model Leaderboard")
            rows = []
            for name, r in results.items():
                rep = r["report"]
                rows.append({
                    "Model": name,
                    "Accuracy": f"{r['acc']:.4f}",
                    "AUC-ROC": f"{r['auc']:.4f}" if r["auc"] else "N/A",
                    "Precision (macro)": f"{rep['macro avg']['precision']:.4f}",
                    "Recall (macro)": f"{rep['macro avg']['recall']:.4f}",
                    "F1 (macro)": f"{rep['macro avg']['f1-score']:.4f}",
                })
            lb_df = pd.DataFrame(rows).set_index("Model")
            st.dataframe(lb_df, width="stretch")

            st.markdown("<br>", unsafe_allow_html=True)
            cols = st.columns(len(models_selected))
            for col, (name, r) in zip(cols, results.items()):
                with col:
                    auc_str = f"{r['auc']:.3f}" if r["auc"] else "N/A"
                    st.markdown(f"""
                    <div style="background:#12121a; border:1px solid #2d2d3d;
                    border-radius:12px; padding:1.2rem; text-align:center;">
                        <div style="font-family:'Space Mono',monospace; font-size:0.7rem;
                        color:#7c3aed; letter-spacing:0.1em; text-transform:uppercase;">
                        {name}</div>
                        <div style="font-family:'Space Mono',monospace; font-size:2.2rem;
                        color:#06b6d4; font-weight:700; margin:0.4rem 0;">
                        {r['acc']:.1%}</div>
                        <div style="color:#64748b; font-size:0.8rem;">Accuracy</div>
                        <div style="color:#f59e0b; font-size:1rem;
                        font-family:'Space Mono',monospace; margin-top:0.4rem;">
                        AUC {auc_str}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # TAB 2 — Detailed Metrics
        with tab2:
            for name, r in results.items():
                st.markdown(f"#### 🤖 {name}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Classification Report**")
                    rep_df = pd.DataFrame(r["report"]).transpose().round(3)
                    st.dataframe(rep_df, width="stretch")
                with c2:
                    st.markdown("**Confusion Matrix**")
                    cm_df = pd.DataFrame(
                        r["cm"],
                        index=[f"Actual {c}" for c in target_classes],
                        columns=[f"Pred {c}" for c in target_classes]
                    )
                    st.dataframe(cm_df, width="stretch")
                st.divider()

        # TAB 3 — Charts
with tab3:

    # ROC Curve
    if n_classes == 2:
        st.markdown("### ROC Curves")

        fig, ax = make_dark_fig()
        colors = ["#7c3aed", "#06b6d4", "#f59e0b"]

        for (name, r), color in zip(results.items(), colors):
            if r["fpr"] is not None:
                ax.plot(
                    r["fpr"],
                    r["tpr"],
                    color=color,
                    linewidth=2,
                    label=f"{name} (AUC={r['auc']:.3f})"
                )

        ax.plot([0, 1], [0, 1],
                color="#2d2d3d",
                linestyle="--",
                linewidth=1)

        ax.set_xlabel("False Positive Rate", color="#94a3b8")
        ax.set_ylabel("True Positive Rate", color="#94a3b8")
        ax.set_title("ROC Curve", color="#e2e8f0", fontsize=12)

        legend = ax.legend(
            facecolor="#12121a",
            edgecolor="#2d2d3d",
            labelcolor="#e2e8f0"
        )

        st.pyplot(fig)
        plt.close()

    # Accuracy Comparison
    st.markdown("### Accuracy Comparison")

    names = list(results.keys())
    accs = [results[n]["acc"] for n in names]

    fig2, ax2 = make_dark_fig()
    bars = ax2.bar(
        names,
        accs,
        color=["#7c3aed", "#06b6d4", "#f59e0b"][:len(names)],
        edgecolor="none",
        alpha=0.9
    )

    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Accuracy", color="#94a3b8")
    ax2.set_title("Model Accuracy", color="#e2e8f0", fontsize=12)

    for bar, acc in zip(bars, accs):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            color="#e2e8f0",
            fontsize=10,
            fontfamily="monospace"
        )

    ax2.tick_params(axis="x", colors="#94a3b8")

    st.pyplot(fig2)
    plt.close()

        # TAB 4 — Feature Importance
    with tab4:
            st.markdown("### Top Feature Importances")
            top_n = min(20, len(X.columns))
            for name, r in results.items():
                m = r["model"]
                if hasattr(m, "feature_importances_"):
                    fi = pd.Series(m.feature_importances_, index=X.columns)
                    fi = fi.nlargest(top_n).sort_values()

                    fig3, ax3 = make_dark_fig()
                    fig3.set_size_inches(7, max(4, top_n * 0.35))
                    ax3.set_facecolor("#1a1a26")
                    bars = ax3.barh(fi.index, fi.values,
                                   color="#7c3aed", edgecolor="none", alpha=0.85)
                    ax3.set_xlabel("Importance", color="#94a3b8")
                    ax3.set_title(f"{name} — Feature Importance", color="#e2e8f0", fontsize=11)
                    ax3.tick_params(axis="y", labelsize=8, colors="#94a3b8")
                    st.pyplot(fig3)
                    plt.close()
                elif hasattr(m, "coef_"):
                    coef = pd.Series(np.abs(m.coef_[0]), index=X.columns)
                    coef = coef.nlargest(top_n).sort_values()
                    fig4, ax4 = make_dark_fig()
                    fig4.set_size_inches(7, max(4, top_n * 0.35))
                    ax4.barh(coef.index, coef.values, color="#06b6d4",
                             edgecolor="none", alpha=0.85)
                    ax4.set_xlabel("|Coefficient|", color="#94a3b8")
                    ax4.set_title(f"{name} — Feature Weights", color="#e2e8f0", fontsize=11)
                    ax4.tick_params(axis="y", labelsize=8, colors="#94a3b8")
                    st.pyplot(fig4)
                    plt.close()
                else:
                    st.info(f"{name} does not support feature importance.")
                    
                    

        # ── Best model summary ────────────────────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["acc"])
    best_acc  = results[best_name]["acc"]
      st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a0a2e, #0a1628);
            border: 1px solid #7c3aed;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            text-align: center;
        ">
            <div style="font-family:'Space Mono',monospace; font-size:0.7rem;
            color:#7c3aed; letter-spacing:0.15em; text-transform:uppercase;">
            🏆 Best Performing Model</div>
            <div style="font-family:'Space Mono',monospace; font-size:1.8rem;
            color:#e2e8f0; font-weight:700; margin:0.5rem 0;">{best_name}</div>
            <div style="color:#06b6d4; font-size:1.1rem;">
            Accuracy: <b>{best_acc:.2%}</b></div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#2d2d3d; font-size:0.75rem;
font-family:'Space Mono',monospace; margin-top:3rem; padding-top:1rem;
border-top:1px solid #1e1e2e;">
AutoPredict AI · Built with Streamlit
</div>
""", unsafe_allow_html=True)
