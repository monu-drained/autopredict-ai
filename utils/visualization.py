import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BG       = "#0a0a0f"
SURFACE  = "#12121a"
SURFACE2 = "#1a1a26"
BORDER   = "#2d2d3d"
TEXT     = "#e2e8f0"
MUTED    = "#94a3b8"
PALETTE  = ["#7c3aed","#06b6d4","#f59e0b","#10b981","#ef4444","#f97316"]

def _dark_fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE2)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ["bottom","left"]:
        ax.spines[spine].set_color(BORDER)
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    return fig, ax

def plot_roc_curves(results):
    fig, ax = _dark_fig(7, 5)
    for (name, r), color in zip(results.items(), PALETTE):
        if r.get("fpr") is not None:
            ax.plot(r["fpr"], r["tpr"], color=color, lw=2, label=f"{name} (AUC={r['auc']:.3f})")
    ax.plot([0,1],[0,1], color=BORDER, ls="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, name):
    fig, ax = _dark_fig(5, 4)
    im = ax.imshow(cm, cmap="Purples")
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha="center", va="center", color=TEXT, fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig

def plot_accuracy_bar(results):
    names = list(results.keys())
    accs = [results[n]["acc"] for n in names]
    fig, ax = _dark_fig(max(5, len(names)*1.5), 4)
    bars = ax.bar(names, accs, color=PALETTE[:len(names)], width=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{acc:.3f}", ha="center", color=TEXT, fontsize=9)
    plt.tight_layout()
    return fig

def plot_regression_metrics(results):
    names = list(results.keys())
    r2s = [results[n]["r2"] for n in names]
    fig, ax = _dark_fig(max(5, len(names)*1.5), 4)
    bars = ax.bar(names, r2s, color=PALETTE[:len(names)], width=0.5)
    ax.set_ylabel("R² Score")
    ax.set_title("R² Score Comparison")
    for bar, v in zip(bars, r2s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{v:.3f}", ha="center", color=TEXT, fontsize=9)
    plt.tight_layout()
    return fig

def plot_pred_vs_actual(y_test, y_pred, name):
    fig, ax = _dark_fig(5, 5)
    ax.scatter(y_test, y_pred, alpha=0.5, color=PALETTE[0], s=20)
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo,hi],[lo,hi], color=PALETTE[1], ls="--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual — {name}")
    plt.tight_layout()
    return fig

def plot_residuals(y_test, y_pred, name):
    residuals = np.array(y_test) - np.array(y_pred)
    fig, ax = _dark_fig(6, 4)
    ax.scatter(y_pred, residuals, alpha=0.5, color=PALETTE[2], s=20)
    ax.axhline(0, color=PALETTE[1], ls="--", lw=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residual Plot — {name}")
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, name, top_n=20):
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_names).nlargest(top_n).sort_values()
        color = PALETTE[0]
        label = "Importance"
    elif hasattr(model, "coef_"):
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        fi = pd.Series(np.abs(coef), index=feature_names).nlargest(top_n).sort_values()
        color = PALETTE[1]
        label = "|Coefficient|"
    else:
        return None
    fig, ax = _dark_fig(7, max(4, top_n*0.35))
    ax.barh(fi.index, fi.values, color=color)
    ax.set_xlabel(label)
    ax.set_title(f"{name} — Feature Importance (Top {top_n})")
    plt.tight_layout()
    return fig

def plot_distribution(series, col_name):
    fig, ax = _dark_fig(6, 3)
    if pd.api.types.is_numeric_dtype(series):
        ax.hist(series.dropna(), bins=30, color=PALETTE[0], edgecolor=BORDER)
        ax.set_xlabel(col_name)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution — {col_name}")
    else:
        vc = series.value_counts().head(15)
        ax.bar(vc.index.astype(str), vc.values, color=PALETTE[0])
        ax.set_xlabel(col_name)
        ax.set_ylabel("Count")
        ax.set_title(f"Value Counts — {col_name}")
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr()
    n = corr.shape[0]
    fig, ax = _dark_fig(max(6, n*0.6), max(5, n*0.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Correlation Heatmap")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig
