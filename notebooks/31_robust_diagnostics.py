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

output_scatter = project_root / "data" / "actual_vs_predicted_central.png"
output_distribution = project_root / "data" / "prediction_distribution_central.png"
output_residual = project_root / "data" / "residual_distribution_central.png"

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
# 6. Train HGB
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

test["Residual"] = test[target] - test["Prediction"]

# =========================
# 8. Basic statistics
# =========================
actual_mean = test[target].mean()
actual_std = test[target].std()

prediction_mean = test["Prediction"].mean()
prediction_std = test["Prediction"].std()

correlation = test[[target, "Prediction"]].corr().iloc[0, 1]

print("=" * 60)
print("Robust Diagnostic Statistics")
print("=" * 60)

print(f"Actual mean:       {actual_mean:.6f}")
print(f"Actual std:        {actual_std:.6f}")

print(f"Prediction mean:   {prediction_mean:.6f}")
print(f"Prediction std:    {prediction_std:.6f}")

print(f"Actual-Prediction correlation: {correlation:.6f}")

# =========================
# 9. Central 98% range
# =========================
actual_low = test[target].quantile(0.01)
actual_high = test[target].quantile(0.99)

prediction_low = test["Prediction"].quantile(0.01)
prediction_high = test["Prediction"].quantile(0.99)

residual_low = test["Residual"].quantile(0.01)
residual_high = test["Residual"].quantile(0.99)

# =========================
# 10. Actual vs Prediction
# =========================
scatter_data = test[
    (test[target] >= actual_low)
    & (test[target] <= actual_high)
    & (test["Prediction"] >= prediction_low)
    & (test["Prediction"] <= prediction_high)
].copy()

sample_size = min(50000, len(scatter_data))

sample = scatter_data.sample(n=sample_size, random_state=42)

plt.figure(figsize=(8, 8))

plt.scatter(sample[target], sample["Prediction"], alpha=0.15, s=8)

lower = min(sample[target].min(), sample["Prediction"].min())

upper = max(sample[target].max(), sample["Prediction"].max())

plt.plot([lower, upper], [lower, upper], linestyle="--")

plt.xlabel("Actual Future 5D Return")

plt.ylabel("Predicted Future 5D Return")

plt.title("Actual vs Predicted Future 5D Return\nCentral 98%")

plt.tight_layout()

plt.savefig(output_scatter, dpi=200)

plt.close()

# =========================
# 11. Distribution
# =========================
actual_central = test[(test[target] >= actual_low) & (test[target] <= actual_high)][
    target
]

prediction_central = test[
    (test["Prediction"] >= prediction_low) & (test["Prediction"] <= prediction_high)
]["Prediction"]

plt.figure(figsize=(10, 6))

plt.hist(actual_central, bins=80, alpha=0.5, label="Actual")

plt.hist(prediction_central, bins=80, alpha=0.5, label="Prediction")

plt.xlabel("Future 5D Return")

plt.ylabel("Frequency")

plt.title("Actual vs Predicted Return Distribution\nCentral 98%")

plt.legend()

plt.tight_layout()

plt.savefig(output_distribution, dpi=200)

plt.close()

# =========================
# 12. Residual distribution
# =========================
residual_central = test[
    (test["Residual"] >= residual_low) & (test["Residual"] <= residual_high)
]["Residual"]

plt.figure(figsize=(10, 6))

plt.hist(residual_central, bins=80)

plt.axvline(0, linestyle="--")

plt.xlabel("Residual (Actual - Predicted)")

plt.ylabel("Frequency")

plt.title("Residual Distribution\nCentral 98%")

plt.tight_layout()

plt.savefig(output_residual, dpi=200)

plt.close()

# =========================
# 13. Output
# =========================
print("\nCentral 98% ranges:")

print(f"Actual:      {actual_low:.6f} to {actual_high:.6f}")

print(f"Prediction:  {prediction_low:.6f} to {prediction_high:.6f}")

print(f"Residual:    {residual_low:.6f} to {residual_high:.6f}")

print("\nSaved figures:")

print(output_scatter)
print(output_distribution)
print(output_residual)
