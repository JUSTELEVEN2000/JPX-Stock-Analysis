import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# 1. File paths
# =========================
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "stock_model_expanded.csv"
output_file = project_root / "data" / "group_baseline_comparison.csv"

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
# 6. Train HGB model
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
# 7. Generate predictions
# =========================
test["Prediction"] = model.predict(X_test)

# =========================
# 8. Overall test baseline
# =========================
overall_baseline = y_train.mean()

test["Baseline_Prediction"] = overall_baseline

test["Model_Error"] = (test[target] - test["Prediction"]).abs()

test["Baseline_Error"] = (test[target] - test["Baseline_Prediction"]).abs()

# =========================
# 9. Function for group analysis
# =========================
results = []

group_columns = [
    "市場・商品区分",
    "33業種区分",
    "規模区分",
]

for group_col in group_columns:

    # Training-period group mean
    group_mean = train.groupby(group_col)[target].mean().rename("Group_Baseline")

    # Map training group mean to test data
    test_group = test.copy()

    test_group = test_group.merge(
        group_mean, left_on=group_col, right_index=True, how="left"
    )

    # Baseline error
    test_group["Group_Baseline_Error"] = (
        test_group[target] - test_group["Group_Baseline"]
    ).abs()

    # Model error
    test_group["Model_Error"] = (test_group[target] - test_group["Prediction"]).abs()

    # Metrics
    result = (
        test_group.groupby(group_col)
        .agg(
            Model_MAE=("Model_Error", "mean"),
            Group_Baseline_MAE=("Group_Baseline_Error", "mean"),
            Target_Std=(target, "std"),
            Target_Mean=(target, "mean"),
            Count=(target, "count"),
        )
        .reset_index()
    )

    result["MAE_Improvement"] = result["Group_Baseline_MAE"] - result["Model_MAE"]

    result["Relative_Improvement"] = (
        result["MAE_Improvement"] / result["Group_Baseline_MAE"]
    )

    result["Group_Type"] = group_col

    results.append(result)

# =========================
# 10. Combine results
# =========================
final_result = pd.concat(results, ignore_index=True)

# =========================
# 11. Save
# =========================
final_result.to_csv(output_file, index=False, encoding="utf-8-sig")

# =========================
# 12. Print results
# =========================
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 200)

for group_type in group_columns:

    print("\n" + "=" * 60)
    print(f"Comparison by {group_type}")
    print("=" * 60)

    result = final_result[final_result["Group_Type"] == group_type].copy()

    print(
        result[
            [
                group_type,
                "Model_MAE",
                "Group_Baseline_MAE",
                "MAE_Improvement",
                "Relative_Improvement",
                "Target_Std",
                "Target_Mean",
                "Count",
            ]
        ].to_string(index=False)
    )

print("\n" + "=" * 60)
print("Overall Test Performance")
print("=" * 60)

print(
    f"Overall baseline MAE: {mean_absolute_error(y_test, np.full(len(y_test), overall_baseline)):.6f}"
)
print(f"HGB model MAE:        {mean_absolute_error(y_test, test['Prediction']):.6f}")

print("\nSaved to:")
print(output_file)
