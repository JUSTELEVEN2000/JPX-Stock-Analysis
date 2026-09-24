import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor

# =========================
# 1. Paths
# =========================
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "stock_model_expanded.csv"

output_actual_predicted = project_root / "data" / "actual_vs_predicted.png"
output_prediction_distribution = project_root / "data" / "prediction_distribution.png"
output_residual_distribution = project_root / "data" / "residual_distribution.png"
output_model_vs_baseline = project_root / "data" / "model_vs_baseline.png"

# =========================
# 2. Load data
# =========================
df = pd.read_csv(input_file, parse_dates=["Date"])

numeric_features = [
    "Return_1D",
    "Return_5D",
    "Return_20D",
    "Return_60D",
    "Volatility_5D",
    "Volatility_20D",
    "Volatility_60D",
    "Volume_Change_5D",
    "Volume_Change_20D",
]

categorical_features = [
    "市場・商品区分",
    "33業種区分",
    "規模区分",
]

target = "Future_5D_Return"

# =========================
# 3. Train / Test split
# =========================
train = df[df["Date"] < "2026-01-01"].copy()
test = df[df["Date"] >= "2026-01-01"].copy()

# =========================
# 4. Encode categorical variables
# =========================
for col in categorical_features:
    categories = pd.Categorical(train[col]).categories

    train[col] = pd.Categorical(train[col], categories=categories).codes

    test[col] = pd.Categorical(test[col], categories=categories).codes

    train.loc[train[col] < 0, col] = np.nan
    test.loc[test[col] < 0, col] = np.nan

# =========================
# 5. Prepare model data
# =========================
features = numeric_features + categorical_features

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# =========================
# 6. Train model
# =========================
model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42,
    categorical_features=[False] * 9 + [True] * 3,
)

model.fit(X_train, y_train)

# =========================
# 7. Predictions
# =========================
test["Prediction"] = model.predict(X_test)

# =========================
# 8. Baseline
# =========================
baseline_prediction = y_train.mean()

test["Baseline_Prediction"] = baseline_prediction

# =========================
# 9. Residual
# =========================
test["Residual"] = test[target] - test["Prediction"]

# =========================================================
# Figure 1: Actual vs Predicted
# =========================================================

# Random sample for visualization
sample_size = min(50000, len(test))

sample = test.sample(n=sample_size, random_state=42)

plt.figure(figsize=(8, 8))

plt.scatter(sample[target], sample["Prediction"], alpha=0.15, s=8)

# Reference line
min_value = min(sample[target].min(), sample["Prediction"].min())

max_value = max(sample[target].max(), sample["Prediction"].max())

plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

plt.xlabel("Actual Future 5D Return")
plt.ylabel("Predicted Future 5D Return")
plt.title("Actual vs Predicted Future 5D Return")

plt.tight_layout()
plt.savefig(output_actual_predicted, dpi=200)

plt.close()

# =========================================================
# Figure 2: Prediction Distribution
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(y_test, bins=100, alpha=0.5, label="Actual")

plt.hist(test["Prediction"], bins=100, alpha=0.5, label="Prediction")

plt.xlabel("Future 5D Return")
plt.ylabel("Frequency")
plt.title("Actual vs Predicted Return Distribution")
plt.legend()

plt.tight_layout()
plt.savefig(output_prediction_distribution, dpi=200)

plt.close()

# =========================================================
# Figure 3: Residual Distribution
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(test["Residual"], bins=100)

plt.axvline(0, linestyle="--")

plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")

plt.tight_layout()
plt.savefig(output_residual_distribution, dpi=200)

plt.close()

# =========================================================
# Figure 4: Model vs Baseline
# =========================================================

model_mae = np.mean(np.abs(test[target] - test["Prediction"]))

baseline_mae = np.mean(np.abs(test[target] - test["Baseline_Prediction"]))

model_rmse = np.sqrt(np.mean((test[target] - test["Prediction"]) ** 2))

baseline_rmse = np.sqrt(np.mean((test[target] - test["Baseline_Prediction"]) ** 2))

labels = ["Baseline", "HGB"]

mae_values = [baseline_mae, model_mae]

rmse_values = [baseline_rmse, model_rmse]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 6))

plt.bar(x - width / 2, mae_values, width, label="MAE")

plt.bar(x + width / 2, rmse_values, width, label="RMSE")

plt.xticks(x, labels)

plt.ylabel("Error")
plt.title("Model vs Baseline")

plt.legend()

plt.tight_layout()
plt.savefig(output_model_vs_baseline, dpi=200)

plt.close()

# =========================
# 10. Print results
# =========================

print("=" * 60)
print("Model Diagnostic Summary")
print("=" * 60)

print(f"Test observations: {len(test):,}")

print("\nMAE")
print(f"Baseline: {baseline_mae:.6f}")
print(f"HGB:      {model_mae:.6f}")

print("\nRMSE")
print(f"Baseline: {baseline_rmse:.6f}")
print(f"HGB:      {model_rmse:.6f}")

print("\nSaved figures:")

print(output_actual_predicted)
print(output_prediction_distribution)
print(output_residual_distribution)
print(output_model_vs_baseline)
