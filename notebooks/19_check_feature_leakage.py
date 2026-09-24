import pandas as pd
import numpy as np
from pathlib import Path

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

# ============================================================
# 2. 排序
# ============================================================

df = df.sort_values(["Ticker", "Date"]).copy()

# ============================================================
# 3. 特征列表
# ============================================================

features = ["Return_5D", "Return_20D", "Volatility_20D", "Volume_Change_5D"]

target = "Future_5D_Return"

# ============================================================
# 4. 基本检查
# ============================================================

print("\n========================================")
print("数据基本信息")
print("========================================")

print("观察值：", len(df))
print("股票数：", df["Ticker"].nunique())
print("交易日：", df["Date"].nunique())

# ============================================================
# 5. Feature / Target 缺失情况
# ============================================================

print("\n========================================")
print("Feature / Target 缺失值")
print("========================================")

check_columns = features + [target]

print(df[check_columns].isna().sum())

# ============================================================
# 6. 检查 Feature 与 Future Target 的关系
# ============================================================

print("\n========================================")
print("Feature 与 Future Target 相关系数")
print("========================================")

correlation = df[features + [target]].corr()[target].sort_values(ascending=False)

print(correlation)

# ============================================================
# 7. 检查最重要的时间逻辑
# ============================================================

print("\n========================================")
print("时间逻辑检查")
print("========================================")

sample = df[df["Ticker"] == "1301.T"].sort_values("Date").head(30)

print(
    sample[
        [
            "Date",
            "Return",
            "Return_5D",
            "Return_20D",
            "Volatility_20D",
            "Volume_Change_5D",
            "Future_5D_Return",
        ]
    ].to_string(index=False)
)

# ============================================================
# 8. 检查是否存在无穷值
# ============================================================

print("\n========================================")
print("Infinite Value Check")
print("========================================")

inf_count = np.isinf(df[features]).sum()

print(inf_count)

print("\nTotal infinite values：", inf_count.sum().sum())

# ============================================================
# 9. 输出结论
# ============================================================

if inf_count.sum().sum() == 0:

    print("\n✓ 没有 Infinite Values")

else:

    print("\n⚠ 仍然存在 Infinite Values")

print("\nFeature Leakage Check 完成！")
