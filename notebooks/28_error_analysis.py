import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# 0. 路径
# ============================================================

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "stock_model_expanded.csv"

output_file = project_root / "data" / "error_analysis.csv"


# ============================================================
# 1. 读取
# ============================================================

print("正在读取数据...")

df = pd.read_csv(input_file, parse_dates=["Date"])

print("数据形状：", df.shape)


# ============================================================
# 2. Feature
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

train = df[df["Date"] < "2026-01-01"].copy()

test = df[df["Date"] >= "2026-01-01"].copy()


# ============================================================
# 4. Categorical Encoding
# ============================================================

for col in categorical_features:

    categories = pd.Categorical(train[col]).categories

    train[col] = pd.Categorical(train[col], categories=categories).codes

    test[col] = pd.Categorical(test[col], categories=categories).codes

    train.loc[train[col] < 0, col] = np.nan

    test.loc[test[col] < 0, col] = np.nan


# ============================================================
# 5. X / y
# ============================================================

features = numeric_features + categorical_features

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


# ============================================================
# 6. Model
# ============================================================

print("\n正在训练模型...")

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

model.fit(X_train, y_train)

print("训练完成！")


# ============================================================
# 7. Prediction
# ============================================================

print("正在预测...")

test["Prediction"] = model.predict(X_test)

test["Error"] = test[target] - test["Prediction"]

test["Absolute_Error"] = test["Error"].abs()

test["Squared_Error"] = test["Error"] ** 2


# ============================================================
# 8. Overall
# ============================================================

print("\n========================================")
print("Overall Test Performance")
print("========================================")

print("MAE：", mean_absolute_error(test[target], test["Prediction"]))

print("RMSE：", np.sqrt(mean_squared_error(test[target], test["Prediction"])))


# ============================================================
# 9. 按市场区分
# ============================================================

print("\n========================================")
print("Performance by Market Segment")
print("========================================")

market_result = (
    test.groupby("市場・商品区分")
    .agg(
        MAE=("Absolute_Error", "mean"),
        RMSE=("Squared_Error", lambda x: np.sqrt(x.mean())),
        Mean_Target=(target, "mean"),
        Count=(target, "count"),
    )
    .reset_index()
)

print(market_result.to_string(index=False))


# ============================================================
# 10. 按行业
# ============================================================

print("\n========================================")
print("Performance by Industry")
print("========================================")

industry_result = (
    test.groupby("33業種区分")
    .agg(
        MAE=("Absolute_Error", "mean"),
        RMSE=("Squared_Error", lambda x: np.sqrt(x.mean())),
        Mean_Target=(target, "mean"),
        Count=(target, "count"),
    )
    .reset_index()
    .sort_values("MAE")
)

print(industry_result.to_string(index=False))


# ============================================================
# 11. 按规模
# ============================================================

print("\n========================================")
print("Performance by Size")
print("========================================")

size_result = (
    test.groupby("規模区分")
    .agg(
        MAE=("Absolute_Error", "mean"),
        RMSE=("Squared_Error", lambda x: np.sqrt(x.mean())),
        Mean_Target=(target, "mean"),
        Count=(target, "count"),
    )
    .reset_index()
)

print(size_result.to_string(index=False))


# ============================================================
# 12. 保存
# ============================================================

market_result["Group_Type"] = "Market"

market_result["Group"] = market_result["市場・商品区分"]

industry_result["Group_Type"] = "Industry"

industry_result["Group"] = industry_result["33業種区分"]

size_result["Group_Type"] = "Size"

size_result["Group"] = size_result["規模区分"]


result = pd.concat(
    [
        market_result[["Group_Type", "Group", "MAE", "RMSE", "Mean_Target", "Count"]],
        industry_result[["Group_Type", "Group", "MAE", "RMSE", "Mean_Target", "Count"]],
        size_result[["Group_Type", "Group", "MAE", "RMSE", "Mean_Target", "Count"]],
    ],
    ignore_index=True,
)

result.to_csv(output_file, index=False)


print("\n========================================")
print("Error Analysis 完成")
print("========================================")

print("结果保存：", output_file)
