import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 0. 路径
# ============================================================

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "stock_model_expanded.csv"

output_file = project_root / "data" / "expanded_gradient_boosting_results.csv"


# ============================================================
# 1. 读取数据
# ============================================================

print("正在读取 Expanded Model Dataset...")

df = pd.read_csv(input_file, parse_dates=["Date"])

print("读取完成！")
print("数据形状：", df.shape)


# ============================================================
# 2. Feature 定义
# ============================================================

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


# ============================================================
# 3. Time Split
# ============================================================

print("\n========================================")
print("Time Split")
print("========================================")

train_mask = df["Date"] < "2026-01-01"

test_mask = df["Date"] >= "2026-01-01"

train = df.loc[train_mask].copy()
test = df.loc[test_mask].copy()

print("Train：", train["Date"].min(), "→", train["Date"].max())

print("Test：", test["Date"].min(), "→", test["Date"].max())

print("Train observations：", len(train))

print("Test observations：", len(test))


# ============================================================
# 4. Categorical Encoding
# ============================================================

print("\n========================================")
print("Categorical Encoding")
print("========================================")

for col in categorical_features:

    categories = pd.Categorical(train[col]).categories

    train[col] = pd.Categorical(train[col], categories=categories).codes

    test[col] = pd.Categorical(test[col], categories=categories).codes

    # 未知类别会变成 -1
    # HGB categorical feature 不适合负数，
    # 所以将未知类别设为 NaN
    train.loc[train[col] < 0, col] = np.nan

    test.loc[test[col] < 0, col] = np.nan


print("Categorical encoding 完成")


# ============================================================
# 5. X / y
# ============================================================

features = numeric_features + categorical_features

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


# ============================================================
# 6. Baseline
# ============================================================

print("\n========================================")
print("Baseline Model")
print("========================================")

baseline_prediction = y_train.mean()

baseline_pred = np.full(len(y_test), baseline_prediction)

baseline_mae = mean_absolute_error(y_test, baseline_pred)

baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

baseline_r2 = r2_score(y_test, baseline_pred)

print("Training target mean：", baseline_prediction)

print("MAE：", baseline_mae)

print("RMSE：", baseline_rmse)

print("R²：", baseline_r2)


# ============================================================
# 7. HistGradientBoosting
# ============================================================

print("\n========================================")
print("Training Expanded HistGradientBoosting...")
print("========================================")

model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42,
    categorical_features=[
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ],
)


# ============================================================
# 8. Training
# ============================================================

model.fit(X_train, y_train)

print("模型训练完成！")


# ============================================================
# 9. Prediction
# ============================================================

print("\n正在预测 Test Set...")

y_pred = model.predict(X_test)


# ============================================================
# 10. Evaluation
# ============================================================

gb_mae = mean_absolute_error(y_test, y_pred)

gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

gb_r2 = r2_score(y_test, y_pred)


print("\n========================================")
print("Expanded HistGradientBoosting Results")
print("========================================")

print("MAE：", gb_mae)

print("RMSE：", gb_rmse)

print("R²：", gb_r2)


# ============================================================
# 11. Comparison
# ============================================================

print("\n========================================")
print("Comparison")
print("========================================")

print("Baseline MAE：", baseline_mae)

print("Gradient Boosting MAE：", gb_mae)

print()

print("Baseline RMSE：", baseline_rmse)

print("Gradient Boosting RMSE：", gb_rmse)

print()

print("Baseline R²：", baseline_r2)

print("Gradient Boosting R²：", gb_r2)


# ============================================================
# 12. 保存结果
# ============================================================

results = pd.DataFrame(
    {
        "Model": ["Baseline", "Expanded HistGradientBoosting"],
        "MAE": [baseline_mae, gb_mae],
        "RMSE": [baseline_rmse, gb_rmse],
        "R2": [baseline_r2, gb_r2],
    }
)

results.to_csv(output_file, index=False)


print("\n========================================")
print("模型结果已保存")
print("========================================")

print(output_file)
