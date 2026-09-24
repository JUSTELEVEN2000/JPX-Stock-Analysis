import pandas as pd
from pathlib import Path

# ============================================================
# 0. 项目路径
# ============================================================

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "stock_analysis_dataset.csv"

output_file = project_root / "data" / "stock_model_dataset.csv"

# ============================================================
# 1. 读取数据
# ============================================================

print("正在读取分析数据...")

df = pd.read_csv(input_file, parse_dates=["Date"])

print("读取完成！")

print("原始数据形状：", df.shape)

# ============================================================
# 2. 排序
# ============================================================

# 非常重要：
# 必须按照 股票 + 日期 排序，
# 才能正确计算每只股票自己的未来收益率。

df = df.sort_values(["Ticker", "Date"]).copy()

# ============================================================
# 3. 构造未来5个交易日累计收益率
# ============================================================

print("\n正在计算 Future 5-Day Return...")

# 每只股票分别计算
#
# (1+r1) × (1+r2) × ... × (1+r5) - 1
#
# shift(-1) 表示从“明天”开始，
# 而不是把今天的收益率算进去。

df["Future_5D_Return"] = df.groupby("Ticker")["Return"].transform(
    lambda x: (
        (1 + x)
        .shift(-1)
        .iloc[::-1]
        .rolling(5)
        .apply(lambda y: y.prod(), raw=True)
        .iloc[::-1]
        - 1
    )
)

# ============================================================
# 4. 检查结果
# ============================================================

print("\nFuture 5-Day Return 示例：")

print(df[["Date", "Ticker", "Return", "Future_5D_Return"]].head(20))

# ============================================================
# 5. 缺失值检查
# ============================================================

missing_future_return = df["Future_5D_Return"].isna().sum()

print("\nFuture 5-Day Return 缺失值：")
print(missing_future_return)

print("缺失比例：", round(missing_future_return / len(df) * 100, 4), "%")

# ============================================================
# 6. 描述统计
# ============================================================

print("\nFuture 5-Day Return 描述统计：")

print(df["Future_5D_Return"].describe())

# ============================================================
# 7. 保存
# ============================================================

df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n========================================")
print("Future Return 数据集创建完成！")
print("========================================")

print("保存位置：")
print(output_file)

print("\n数据形状：")
print(df.shape)
