import pandas as pd
import numpy as np

from pathlib import Path

# ============================================================
# 0. 路径
# ============================================================

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "stock_feature_dataset.csv"

output_file = project_root / "data" / "stock_feature_expanded.csv"


# ============================================================
# 1. 读取数据
# ============================================================

print("正在读取 Feature Dataset...")

df = pd.read_csv(input_file, parse_dates=["Date"])

print("读取完成！")
print("原始数据形状：", df.shape)


# ============================================================
# 2. 排序
# ============================================================

print("\n正在按照股票和日期排序...")

df = df.sort_values(["Ticker", "Date"]).copy()


# ============================================================
# 3. Return_1D
# ============================================================

print("\n正在计算 Return_1D...")

df["Return_1D"] = df["Return"]


# ============================================================
# 4. Return_60D
# ============================================================

print("正在计算 Return_60D...")

df["Return_60D"] = df.groupby("Ticker")["Return"].transform(
    lambda x: (1 + x).rolling(60).apply(np.prod, raw=True) - 1
)


# ============================================================
# 5. Volatility_5D
# ============================================================

print("正在计算 Volatility_5D...")

df["Volatility_5D"] = df.groupby("Ticker")["Return"].transform(
    lambda x: x.rolling(5).std()
)


# ============================================================
# 6. Volatility_60D
# ============================================================

print("正在计算 Volatility_60D...")

df["Volatility_60D"] = df.groupby("Ticker")["Return"].transform(
    lambda x: x.rolling(60).std()
)


# ============================================================
# 7. Volume_Change_20D
# ============================================================

print("正在计算 Volume_Change_20D...")

volume_mean_20d = df.groupby("Ticker")["Volume"].transform(
    lambda x: x.shift(1).rolling(20).mean()
)

df["Volume_Change_20D"] = df["Volume"] / volume_mean_20d - 1

# 处理除以 0 产生的 inf
df["Volume_Change_20D"] = df["Volume_Change_20D"].replace([np.inf, -np.inf], np.nan)


# ============================================================
# 8. 新 Feature 列表
# ============================================================

new_features = [
    "Return_1D",
    "Return_60D",
    "Volatility_5D",
    "Volatility_60D",
    "Volume_Change_20D",
]


# ============================================================
# 9. 缺失值检查
# ============================================================

print("\n========================================")
print("新增 Feature 缺失值")
print("========================================")

print(df[new_features].isna().sum())


# ============================================================
# 10. Infinite Value Check
# ============================================================

print("\n========================================")
print("Infinite Value Check")
print("========================================")

inf_count = np.isinf(df[new_features]).sum()

print(inf_count)

print("\nTotal infinite values：", inf_count.sum())


# ============================================================
# 11. 描述统计
# ============================================================

print("\n========================================")
print("新增 Feature 描述统计")
print("========================================")

print(df[new_features].describe())


# ============================================================
# 12. 保存
# ============================================================

df.to_csv(output_file, index=False)

print("\n========================================")
print("Feature Expansion 完成！")
print("========================================")

print("输出文件：")
print(output_file)

print("数据形状：")
print(df.shape)

print("\n新增 Features：")
for feature in new_features:
    print("-", feature)
