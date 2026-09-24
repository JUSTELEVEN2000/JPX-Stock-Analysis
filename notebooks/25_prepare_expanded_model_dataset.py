import pandas as pd
import numpy as np

from pathlib import Path

# ============================================================
# 0. 路径
# ============================================================

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "stock_feature_expanded.csv"

output_file = project_root / "data" / "stock_model_expanded.csv"


# ============================================================
# 1. 读取数据
# ============================================================

print("正在读取 Expanded Feature Dataset...")

df = pd.read_csv(input_file, parse_dates=["Date"])

print("读取完成！")
print("原始数据形状：", df.shape)


# ============================================================
# 2. 定义 Feature
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
# 3. 检查必要列
# ============================================================

required_columns = ["Date", "Ticker", target] + numeric_features + categorical_features

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:

    print("\nERROR：以下列不存在：")

    for col in missing_columns:
        print("-", col)

    raise ValueError("必要な列がありません。")


# ============================================================
# 4. 只保留建模需要的列
# ============================================================

model_df = df[required_columns].copy()


# ============================================================
# 5. 删除 Feature / Target 缺失值
# ============================================================

print("\n正在删除 Feature / Target 缺失值...")

before_rows = len(model_df)

model_df = model_df.dropna(
    subset=(numeric_features + categorical_features + [target])
).copy()

after_rows = len(model_df)

removed_rows = before_rows - after_rows

print("删除前：", before_rows)

print("删除后：", after_rows)

print("删除数量：", removed_rows)

print("保留比例：", f"{after_rows / before_rows:.2%}")


# ============================================================
# 6. 检查 Infinite
# ============================================================

print("\n========================================")
print("Infinite Value Check")
print("========================================")

inf_count = np.isinf(model_df[numeric_features]).sum()

print(inf_count)

print("\nTotal infinite values：", inf_count.sum())


# ============================================================
# 7. 分类变量检查
# ============================================================

print("\n========================================")
print("Categorical Feature Check")
print("========================================")

for col in categorical_features:

    print(f"\n{col}")

    print("类别数量：", model_df[col].nunique())

    print(model_df[col].value_counts().head(15))


# ============================================================
# 8. 日期范围
# ============================================================

print("\n========================================")
print("Date Range")
print("========================================")

print("最早日期：", model_df["Date"].min())

print("最晚日期：", model_df["Date"].max())


# ============================================================
# 9. 股票数量
# ============================================================

print("\n========================================")
print("Stock Count")
print("========================================")

print("股票数量：", model_df["Ticker"].nunique())


# ============================================================
# 10. Target Statistics
# ============================================================

print("\n========================================")
print("Target Statistics")
print("========================================")

print(model_df[target].describe())


# ============================================================
# 11. 排序
# ============================================================

model_df = model_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


# ============================================================
# 12. 保存
# ============================================================

model_df.to_csv(output_file, index=False)

print("\n========================================")
print("Expanded Model Dataset 完成！")
print("========================================")

print("输出文件：", output_file)

print("最终数据形状：", model_df.shape)

print("\nNumeric Features：")

for feature in numeric_features:
    print("-", feature)

print("\nCategorical Features：")

for feature in categorical_features:
    print("-", feature)
