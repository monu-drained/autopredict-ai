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

header[data-testid="stHeader"] {
    background: transparent;
}

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid #1e1e2e;
}

section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #9333ea) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.5rem !important;
}

[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid #2d2d3d !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 4px !important;
}

.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg,#12002f,#090016);
    padding: 2rem;
    border-radius: 24px;
    border: 1px solid rgba(138,43,226,0.25);
    box-shadow: 0 0 30px rgba(138,43,226,0.15);
">

<p style="
    color:#8b5cf6;
    letter-spacing:6px;
    font-size:0.8rem;
    margin-bottom:0.5rem;
">
• MACHINE LEARNING STUDIO
</p>

<h1 style="
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0.4rem 0 0.6rem;
    background: linear-gradient(90deg,#a855f7,#3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">
AutoPredict AI 🔮
</h1>

<p style="color:#94a3b8;">
Upload any CSV → auto-detect target → train ML models → visualize results instantly.
</p>

</div>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────────────────────

def detect_target_column(df):
    keywords = [
        "default", "loan_default", "status", "label", "target",
        "outcome", "churn", "fraud", "result", "class",
        "approved", "survived", "y", "output", "prediction"
    ]

    for col in df.columns:
        if col.lower() in keywords:
            return col

    return df.columns[-1]


def clean_dataframe(df):
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

    id_cols = [
        c for c in df.columns
        if c.lower() in [
            "id", "loan_id", "customerid",
            "customer_id", "userid", "user_id"
        ]
    ]

    X = df.drop(columns=id_cols + [target_col], errors="ignore")
    y = df[target_col].copy()

    if y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_col)
        target_classes = le.classes_
    else:
        y = y.astype(int)
        target_classes = sorted(y.unique())

    X = pd.get_dummies(X, drop_first=True)

    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    return X, y, target_classes


def make_dark_fig():
    fig, ax = plt.subplots(figsize=(6, 4))

    fig.patch.set_facecolor("#12121a")
    ax.set_facecolor("#1a1a26")

    ax.tick_params(colors="#94a3b8")

    ax.spines["bottom"].set_color("#2d2d3d")
    ax.spines["left"].set_color("#2d2d3d")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig, ax


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("### 📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    st.divider()

    st.markdown("### 🎯 Model Settings")

    models_selected = st.multiselect(
        "Choose models",
        [
            "Random Forest",
            "Gradient Boosting",
            "Logistic Regression"
        ],
        default=[
            "Random Forest",
            "Gradient Boosting"
        ]
    )

    test_size = st.slider(
        "Test size",
        0.1,
        0.4,
        0.2,
        0.05
    )


# ── Main ──────────────────────────────────────────────────────────────────────
if uploaded_file is None:

    st.info("👈 Upload a CSV file to begin.")

else:

    try:
        df_raw = pd.read_csv(uploaded_file)

    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    df, dropped_cols = clean_dataframe(df_raw)

    if dropped_cols:
        st.warning(
            f"Dropped columns: {', '.join(dropped_cols)}"
        )

    target_col = detect_target_column(df)

    col1, col2 = st.columns([3, 1])

    with col1:
        target_col = st.selectbox(
            "🎯 Target column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(target_col)
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)

        run_btn = st.button(
            "▶ Run Analysis"
        )

    # Dataset preview
    with st.expander("🔍 Dataset Preview"):

        st.markdown(
            f"Rows: {df_raw.shape[0]} | Columns: {df_raw.shape[1]}"
        )

        st.dataframe(
            df_raw.head(10),
            use_container_width=True
        )

    # Run analysis
    if run_btn or st.session_state.get("ran_once"):

        st.session_state["ran_once"] = True

        if not models_selected:
            st.error("Select at least one model.")
            st.stop()

        n_classes = df[target_col].nunique()

        if n_classes > 20:
            st.error(
                f"Too many target classes ({n_classes})."
            )
            st.stop()

        # Prepare features
        try:
            X, y, target_classes = prepare_features(
                df,
                target_col
            )

        except Exception as e:
            st.error(f"Preparation failed: {e}")
            st.stop()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y if n_classes <= 10 else None
        )

        scaler = StandardScaler()

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Models
        model_map = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced"
            ),

            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),

            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            ),
        }

        results = {}

        progress = st.progress(
            0,
            text="Training models..."
        )

        # Train models
        for i, name in enumerate(models_selected):

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

                    auc = roc_auc_score(
                        y_test,
                        proba,
                        multi_class="ovr"
                    )

                    fpr, tpr = None, None

            except Exception:
                auc, fpr, tpr = None, None, None

            results[name] = {
                "model": m,
                "pred": y_pred,
                "acc": acc,
                "auc": auc,
                "fpr": fpr,
                "tpr": tpr,
                "report": classification_report(
                    y_test,
                    y_pred,
                    output_dict=True
                ),
                "cm": confusion_matrix(
                    y_test,
                    y_pred
                )
            }

            progress.progress(
                (i + 1) / len(models_selected),
                text=f"{name} complete"
            )

        progress.empty()

        st.success(
            f"Successfully trained {len(models_selected)} model(s)"
        )

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview",
            "🔢 Metrics",
            "📊 Charts",
            "🔬 Feature Importance"
        ])

        # TAB 1
        with tab1:

            rows = []

            for name, r in results.items():

                rep = r["report"]

                rows.append({
                    "Model": name,
                    "Accuracy": f"{r['acc']:.4f}",
                    "AUC": f"{r['auc']:.4f}" if r["auc"] else "N/A",
                    "Precision": f"{rep['macro avg']['precision']:.4f}",
                    "Recall": f"{rep['macro avg']['recall']:.4f}",
                    "F1": f"{rep['macro avg']['f1-score']:.4f}",
                })

            lb_df = pd.DataFrame(rows)

            st.dataframe(
                lb_df,
                use_container_width=True
            )

        # TAB 2
        with tab2:

            for name, r in results.items():

                st.markdown(f"### {name}")

                c1, c2 = st.columns(2)

                with c1:

                    rep_df = pd.DataFrame(
                        r["report"]
                    ).transpose().round(3)

                    st.dataframe(
                        rep_df,
                        use_container_width=True
                    )

                with c2:

                    cm_df = pd.DataFrame(
                        r["cm"]
                    )

                    st.dataframe(
                        cm_df,
                        use_container_width=True
                    )

                st.divider()

        # TAB 3
        with tab3:

            if n_classes == 2:

                st.markdown("### ROC Curve")

                fig, ax = make_dark_fig()

                colors = [
                    "#7c3aed",
                    "#06b6d4",
                    "#f59e0b"
                ]

                for (name, r), color in zip(results.items(), colors):

                    if r["fpr"] is not None:

                        ax.plot(
                            r["fpr"],
                            r["tpr"],
                            color=color,
                            linewidth=2,
                            label=f"{name} (AUC={r['auc']:.3f})"
                        )

                ax.plot(
                    [0, 1],
                    [0, 1],
                    color="#2d2d3d",
                    linestyle="--"
                )

                ax.set_xlabel(
                    "False Positive Rate",
                    color="#94a3b8"
                )

                ax.set_ylabel(
                    "True Positive Rate",
                    color="#94a3b8"
                )

                ax.legend()

                st.pyplot(fig)

                plt.close()

            # Accuracy chart
            st.markdown("### Accuracy Comparison")

            names = list(results.keys())

            accs = [
                results[n]["acc"]
                for n in names
            ]

            fig2, ax2 = make_dark_fig()

            bars = ax2.bar(
                names,
                accs,
                color=[
                    "#7c3aed",
                    "#06b6d4",
                    "#f59e0b"
                ][:len(names)]
            )

            ax2.set_ylim(0, 1)

            for bar, acc in zip(bars, accs):

                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{acc:.3f}",
                    ha="center"
                )

            st.pyplot(fig2)

            plt.close()

        # TAB 4
        with tab4:

            st.markdown("### Feature Importance")

            top_n = min(20, len(X.columns))

            for name, r in results.items():

                m = r["model"]

                if hasattr(m, "feature_importances_"):

                    fi = pd.Series(
                        m.feature_importances_,
                        index=X.columns
                    )

                    fi = fi.nlargest(top_n).sort_values()

                    fig3, ax3 = make_dark_fig()

                    fig3.set_size_inches(
                        7,
                        max(4, top_n * 0.35)
                    )

                    ax3.barh(
                        fi.index,
                        fi.values,
                        color="#7c3aed"
                    )

                    ax3.set_title(
                        f"{name} Feature Importance",
                        color="#e2e8f0"
                    )

                    st.pyplot(fig3)

                    plt.close()

                elif hasattr(m, "coef_"):

                    coef = pd.Series(
                        np.abs(m.coef_[0]),
                        index=X.columns
                    )

                    coef = coef.nlargest(top_n).sort_values()

                    fig4, ax4 = make_dark_fig()

                    fig4.set_size_inches(
                        7,
                        max(4, top_n * 0.35)
                    )

                    ax4.barh(
                        coef.index,
                        coef.values,
                        color="#06b6d4"
                    )

                    ax4.set_title(
                        f"{name} Feature Weights",
                        color="#e2e8f0"
                    )

                    st.pyplot(fig4)

                    plt.close()

                else:

                    st.info(
                        f"{name} does not support feature importance."
                    )

        # Best model
        best_name = max(
            results,
            key=lambda n: results[n]["acc"]
        )

        best_acc = results[best_name]["acc"]

        st.markdown(f"""
<div style="background: linear-gradient(135deg, #1a0a2e, #0a1628); border: 1px solid #7c3aed; border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem; text-align: center;">
    <div style="font-family:'Space Mono',monospace; font-size:1.8rem; color:#e2e8f0; font-weight:700;">
        🏆 {best_name}
    </div>
    <div style="color:#06b6d4; font-size:1.1rem;">
        Accuracy: <b>{best_acc:.2%}</b>
    </div>
</div>
""", unsafe_allow_html=True)
# Footer
st.markdown("""
<div style="
    text-align:center;
    color:#2d2d3d;
    font-size:0.75rem;
    margin-top:3rem;
    padding-top:1rem;
    border-top:1px solid #1e1e2e;
">
AutoPredict AI · Built with Streamlit
</div>
""", unsafe_allow_html=True)
