import pandas as pd
import numpy as np

from pathlib import Path

# ============================================================
# 0. 路径
# ============================================================

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "stock_feature_expanded.csv"


# ============================================================
# 1. 读取
# ============================================================

print("正在读取 Expanded Feature Dataset...")

df = pd.read_csv(input_file, parse_dates=["Date"])

print("读取完成！")
print("数据形状：", df.shape)


# ============================================================
# 2. Feature 列表
# ============================================================

features = [
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

target = "Future_5D_Return"


# ============================================================
# 3. 检查 Infinite Values
# ============================================================

print("\n========================================")
print("1. Infinite Value Check")
print("========================================")

inf_count = np.isinf(df[features + [target]]).sum()

print(inf_count)

print("\nTotal infinite values：", inf_count.sum())


# ============================================================
# 4. Missing Values
# ============================================================

print("\n========================================")
print("2. Missing Value Check")
print("========================================")

print(df[features + [target]].isna().sum())


# ============================================================
# 5. Correlation with Future Target
# ============================================================

print("\n========================================")
print("3. Correlation with Future_5D_Return")
print("========================================")

correlation = df[features + [target]].corr()[target].sort_values(ascending=False)

print(correlation)


# ============================================================
# 6. Feature-Target Correlation
# ============================================================

print("\n========================================")
print("Feature → Target Correlation")
print("========================================")

for feature in features:

    corr = df[[feature, target]].corr().iloc[0, 1]

    print(f"{feature:22s}: {corr:.6f}")


# ============================================================
# 7. 检查是否存在异常高相关
# ============================================================

print("\n========================================")
print("4. High Correlation Check")
print("========================================")

for feature in features:

    corr = abs(df[[feature, target]].corr().iloc[0, 1])

    if corr > 0.50:
        print(f"WARNING: {feature} correlation = {corr:.6f}")

print("\n一般来说，如果没有接近 1 的异常相关，" "就没有明显的 target leakage 迹象。")


print("\n========================================")
print("Leakage Check 完成")
print("========================================")
