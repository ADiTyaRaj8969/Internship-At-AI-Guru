import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
# ==== CLASSIFICATION METRICS ====
# Example predictions and true labels
y_true_cls = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred_cls = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
# Accuracy
accuracy = accuracy_score(y_true_cls, y_pred_cls)  # (TP + TN) / total
# Precision
precision = precision_score(y_true_cls, y_pred_cls)  # TP / (TP + FP)
# Recall (Sensitivity)
recall = recall_score(y_true_cls, y_pred_cls)  # TP / (TP + FN)
# F1 Score
f1 = f1_score(y_true_cls, y_pred_cls)  # Harmonic mean of precision & recal
# Confusion Matrix
conf_matrix = confusion_matrix(y_true_cls, y_pred_cls)  # TP, TN, FP, FN layout
# ROC AUC Score
roc_auc = roc_auc_score(y_true_cls, y_pred_cls)
# ==== REGRESSION METRICS ====
# Example predictions and true values
y_true_reg = np.array([3.0, -0.5, 2.0, 7.0])
y_pred_reg = np.array([2.5, 0.0, 2.1, 7.8])
# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_true_reg, y_pred_reg)  # mean(|y_true - y_pred|)
# Mean Squared Error (MSE)
mse = mean_squared_error(y_true_reg, y_pred_reg)  # mean((y_true - y_pred)^2)
# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)  # sqrt(MSE)
# R^2 Score (Coefficient of Determination)
r2 = r2_score(y_true_reg, y_pred_reg)  # 1 - (SS_res / SS_tot)
# ==== DISPLAY RESULTS ====
print("--- Classification Metrics ---")
print(f"Accuracy     : {accuracy:.2f}")
print(f"Precision    : {precision:.2f}")
print(f"Recall       : {recall:.2f}")
print(f"F1 Score     : {f1:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"ROC AUC      : {roc_auc:.2f}")
print("\n--- Regression Metrics ---")
print(f"MAE          : {mae:.2f}")
print(f"MSE          : {mse:.2f}")
print(f"RMSE         : {rmse:.2f}")
print(f"R^2 Score    : {r2:.2f}")
