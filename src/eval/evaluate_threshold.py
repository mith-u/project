import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# Load scores (must include hybrid_score and ground truth labels)
df = pd.read_csv("artifacts/scores.csv")

# ⚠️ IMPORTANT: You need a column "label"
# label = 0 (normal/human), label = 1 (anomaly/bot)
if "label" not in df.columns:
    raise ValueError("scores.csv must contain a 'label' column with ground truth (0=normal, 1=anomaly)")

y_true = df["label"].values
y_scores = df["hybrid_score"].values

# --- ROC Curve ---
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Find best threshold using Youden's J statistic
J = tpr - fpr
best_idx = J.argmax()
best_threshold = thresholds[best_idx]

print(f"Best Threshold (ROC): {best_threshold:.3f}")
print(f"AUC: {roc_auc:.3f}")

# --- Precision-Recall Curve ---
precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
pr_f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
best_pr_idx = pr_f1.argmax()
best_pr_threshold = pr_thresholds[best_pr_idx]

print(f"Best Threshold (PR/F1): {best_pr_threshold:.3f}")
print(f"Best F1 Score: {pr_f1[best_pr_idx]:.3f}")

# --- Plot ROC ---
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("artifacts/reports/roc_curve.png")
plt.close()

# --- Plot Precision-Recall ---
plt.figure()
plt.plot(recall, precision, label="PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.savefig("artifacts/reports/pr_curve.png")
plt.close()

print("Saved ROC and PR plots in artifacts/reports/")
