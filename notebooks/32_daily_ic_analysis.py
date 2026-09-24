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
output_file = project_root / "data" / "daily_ic.csv"
output_plot = project_root / "data" / "daily_ic_distribution.png"

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

# =========================
# 8. Daily Spearman IC
# =========================
daily_ic = (
    test.groupby("Date")
    .apply(
        lambda x: x["Prediction"].corr(x[target], method="spearman"),
        include_groups=False,
    )
    .dropna()
    .reset_index(name="IC")
)

# =========================
# 9. Summary
# =========================
mean_ic = daily_ic["IC"].mean()
median_ic = daily_ic["IC"].median()
std_ic = daily_ic["IC"].std()

positive_ic_ratio = (daily_ic["IC"] > 0).mean()

negative_ic_ratio = (daily_ic["IC"] < 0).mean()

print("=" * 60)
print("Daily Cross-sectional IC Analysis")
print("=" * 60)

print(f"Number of trading days: {len(daily_ic)}")
print(f"Mean IC:                {mean_ic:.6f}")
print(f"Median IC:              {median_ic:.6f}")
print(f"IC Std:                 {std_ic:.6f}")
print(f"Positive IC days:       {positive_ic_ratio:.2%}")
print(f"Negative IC days:       {negative_ic_ratio:.2%}")

print("\nIC Quantiles:")

print(daily_ic["IC"].quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]))

# =========================
# 10. Save daily IC
# =========================
daily_ic.to_csv(output_file, index=False, encoding="utf-8-sig")

# =========================
# 11. Plot IC distribution
# =========================
plt.figure(figsize=(10, 6))

plt.hist(daily_ic["IC"], bins=40)

plt.axvline(0, linestyle="--")

plt.axvline(mean_ic, linestyle="--")

plt.xlabel("Daily Cross-sectional Spearman IC")

plt.ylabel("Frequency")

plt.title("Distribution of Daily Cross-sectional IC")

plt.tight_layout()

plt.savefig(output_plot, dpi=200)

plt.close()

# =========================
# 12. Output
# =========================
print("\nSaved:")
print(output_file)
print(output_plot)
