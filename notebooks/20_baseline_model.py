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
# 2. 定义 Features 和 Target
# ============================================================

features = ["Return_5D", "Return_20D", "Volatility_20D", "Volume_Change_5D"]

target = "Future_5D_Return"


# ============================================================
# 3. 删除 Feature / Target 缺失值
# ============================================================

model_df = df[features + [target, "Date", "Ticker"]].dropna().copy()

print("\n========================================")
print("建模数据")
print("========================================")

print("建模数据形状：", model_df.shape)
print("股票数量：", model_df["Ticker"].nunique())
print("日期范围：")
print(model_df["Date"].min(), "→", model_df["Date"].max())


# ============================================================
# 4. 按时间排序
# ============================================================

model_df = model_df.sort_values("Date").copy()


# ============================================================
# 5. 时间切分
# ============================================================

split_date = pd.Timestamp("2026-01-01")

train = model_df[model_df["Date"] < split_date].copy()

test = model_df[model_df["Date"] >= split_date].copy()

print("\n========================================")
print("Train / Test Split")
print("========================================")

print("Train：")
print(train["Date"].min(), "→", train["Date"].max())
print("Train observations：", len(train))

print("\nTest：")
print(test["Date"].min(), "→", test["Date"].max())
print("Test observations：", len(test))


# ============================================================
# 6. X / y
# ============================================================

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


# ============================================================
# 7. Baseline Model
# ============================================================

print("\n========================================")
print("Baseline Model")
print("========================================")

baseline_prediction = y_train.mean()

baseline_pred = np.full(len(y_test), baseline_prediction)

baseline_mae = mean_absolute_error(y_test, baseline_pred)

baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

baseline_r2 = 1 - (
    np.sum((y_test - baseline_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
)

print("Baseline prediction：", baseline_prediction)
print("Baseline MAE：", baseline_mae)
print("Baseline RMSE：", baseline_rmse)
print("Baseline R²：", baseline_r2)


# ============================================================
# 8. Ridge Regression
# ============================================================

print("\n========================================")
print("Ridge Regression")
print("========================================")

model = Ridge(alpha=1.0)

model.fit(X_train, y_train)

prediction = model.predict(X_test)


# ============================================================
# 9. Model Evaluation
# ============================================================

mae = mean_absolute_error(y_test, prediction)

rmse = np.sqrt(mean_squared_error(y_test, prediction))

r2 = model.score(X_test, y_test)

print("Ridge MAE：", mae)
print("Ridge RMSE：", rmse)
print("Ridge R²：", r2)


# ============================================================
# 10. Feature Coefficients
# ============================================================

print("\n========================================")
print("Feature Coefficients")
print("========================================")

coefficients = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})

print(coefficients.sort_values("Coefficient", ascending=False).to_string(index=False))


# ============================================================
# 11. Model Comparison
# ============================================================

print("\n========================================")
print("Model Comparison")
print("========================================")

comparison = pd.DataFrame(
    {
        "Model": ["Baseline", "Ridge"],
        "MAE": [baseline_mae, mae],
        "RMSE": [baseline_rmse, rmse],
        "R2": [baseline_r2, r2],
    }
)

print(comparison.to_string(index=False))


# ============================================================
# 12. 保存结果
# ============================================================

output_file = project_root / "data" / "baseline_model_results.csv"

comparison.to_csv(output_file, index=False)

print("\n结果已保存：")
print(output_file)

print("\nBaseline Model 完成！")
