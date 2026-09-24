import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# 0. 路径
# ============================================================

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "stock_feature_dataset.csv"


# ============================================================
# 1. 读取数据
# ============================================================

print("正在读取 Feature Dataset...")

df = pd.read_csv(input_file, parse_dates=["Date"])

print("读取完成！")
print("原始数据形状：", df.shape)


# ============================================================
# 2. Features / Target
# ============================================================

features = ["Return_5D", "Return_20D", "Volatility_20D", "Volume_Change_5D"]

target = "Future_5D_Return"


# ============================================================
# 3. 删除缺失值
# ============================================================

model_df = df[features + [target, "Date", "Ticker"]].dropna().copy()

model_df = model_df.sort_values("Date").copy()


# ============================================================
# 4. Train / Test Split
# ============================================================

split_date = pd.Timestamp("2026-01-01")

train = model_df[model_df["Date"] < split_date].copy()

test = model_df[model_df["Date"] >= split_date].copy()


print("\n========================================")
print("Train / Test")
print("========================================")

print("Train：", train["Date"].min(), "→", train["Date"].max())

print("Test：", test["Date"].min(), "→", test["Date"].max())

print("Train observations：", len(train))
print("Test observations：", len(test))


# ============================================================
# 5. 查看原始 Target
# ============================================================

print("\n========================================")
print("Original Target")
print("========================================")

print(train[target].describe())


# ============================================================
# 6. 根据 TRAINING SET 确定极端值阈值
# ============================================================

threshold = 0.50

print("\n========================================")
print("Robustness Threshold")
print("========================================")

print("极端值定义：|Future_5D_Return| >", threshold)


# ============================================================
# 7. 只删除 TRAINING SET 中的极端 Target
# ============================================================

train_robust = train[train[target].abs() <= threshold].copy()

test_robust = test[test[target].abs() <= threshold].copy()


print("\n========================================")
print("Extreme Target Removal")
print("========================================")

print("Train 原始：", len(train))

print("Train 删除极端值后：", len(train_robust))

print("Train 删除数量：", len(train) - len(train_robust))

print("Test 原始：", len(test))

print("Test 删除极端值后：", len(test_robust))

print("Test 删除数量：", len(test) - len(test_robust))


# ============================================================
# 8. 建立 Robust Ridge
# ============================================================

print("\n========================================")
print("Robust Ridge Regression")
print("========================================")

X_train = train_robust[features]
y_train = train_robust[target]

X_test = test_robust[features]
y_test = test_robust[target]


model = Ridge(alpha=1.0)

model.fit(X_train, y_train)

prediction = model.predict(X_test)


# ============================================================
# 9. Robust Baseline
# ============================================================

baseline_prediction = y_train.mean()

baseline_pred = np.full(len(y_test), baseline_prediction)


# ============================================================
# 10. Evaluation Function
# ============================================================


def evaluate_model(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))

    return mae, rmse, r2


# ============================================================
# 11. Evaluate
# ============================================================

baseline_mae, baseline_rmse, baseline_r2 = evaluate_model(y_test, baseline_pred)

ridge_mae, ridge_rmse, ridge_r2 = evaluate_model(y_test, prediction)


# ============================================================
# 12. Results
# ============================================================

print("\n========================================")
print("Robustness Results")
print("========================================")

print("\nBaseline")
print("MAE：", baseline_mae)
print("RMSE：", baseline_rmse)
print("R²：", baseline_r2)

print("\nRidge")
print("MAE：", ridge_mae)
print("RMSE：", ridge_rmse)
print("R²：", ridge_r2)


# ============================================================
# 13. Feature Coefficients
# ============================================================

print("\n========================================")
print("Robust Ridge Coefficients")
print("========================================")

coefficients = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})

print(coefficients.sort_values("Coefficient", ascending=False).to_string(index=False))


# ============================================================
# 14. 保存结果
# ============================================================

comparison = pd.DataFrame(
    {
        "Model": ["Robust Baseline", "Robust Ridge"],
        "MAE": [baseline_mae, ridge_mae],
        "RMSE": [baseline_rmse, ridge_rmse],
        "R2": [baseline_r2, ridge_r2],
    }
)

output_file = project_root / "data" / "model_robustness_results.csv"

comparison.to_csv(output_file, index=False)

print("\n结果已保存：")
print(output_file)

print("\nModel Robustness Check 完成！")
