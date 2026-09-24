import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 0. 路径
# ============================================================

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "stock_model_expanded.csv"

output_file = project_root / "data" / "expanded_ridge_results.csv"


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
# 4. X / y
# ============================================================

X_train = train[numeric_features + categorical_features]

y_train = train[target]

X_test = test[numeric_features + categorical_features]

y_test = test[target]


# ============================================================
# 5. Baseline
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
# 6. Preprocessing
# ============================================================

print("\n========================================")
print("Building Preprocessing Pipeline")
print("========================================")

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# ============================================================
# 7. Ridge Model
# ============================================================

print("\n========================================")
print("Training Standardized Ridge Model")
print("========================================")

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("ridge", Ridge(alpha=1.0)),
    ]
)


model.fit(X_train, y_train)


# ============================================================
# 8. Prediction
# ============================================================

print("正在预测 Test Set...")

y_pred = model.predict(X_test)


# ============================================================
# 9. Evaluation
# ============================================================

ridge_mae = mean_absolute_error(y_test, y_pred)

ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

ridge_r2 = r2_score(y_test, y_pred)


print("\n========================================")
print("Expanded Ridge Results")
print("========================================")

print("MAE：", ridge_mae)

print("RMSE：", ridge_rmse)

print("R²：", ridge_r2)


# ============================================================
# 10. 与 Baseline 比较
# ============================================================

print("\n========================================")
print("Comparison")
print("========================================")

print("Baseline MAE：", baseline_mae)

print("Ridge MAE：", ridge_mae)

print()

print("Baseline RMSE：", baseline_rmse)

print("Ridge RMSE：", ridge_rmse)

print()

print("Baseline R²：", baseline_r2)

print("Ridge R²：", ridge_r2)


# ============================================================
# 11. 保存结果
# ============================================================

results = pd.DataFrame(
    {
        "Model": ["Baseline", "Expanded Ridge"],
        "MAE": [baseline_mae, ridge_mae],
        "RMSE": [baseline_rmse, ridge_rmse],
        "R2": [baseline_r2, ridge_r2],
    }
)

results.to_csv(output_file, index=False)


print("\n========================================")
print("模型结果已保存")
print("========================================")

print(output_file)
