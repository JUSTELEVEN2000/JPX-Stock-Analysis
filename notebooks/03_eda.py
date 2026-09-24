import pandas as pd
from pathlib import Path

# =========================
# 1. 项目路径
# =========================

project_root = Path(__file__).resolve().parent.parent

data_file = project_root / "data" / "stock_analysis_dataset.csv"


# =========================
# 2. 读取数据
# =========================

print("正在读取分析数据...")

df = pd.read_csv(data_file)

print("读取完成！")


# =========================
# 3. 基本信息
# =========================

print("\n" + "=" * 60)
print("1. 数据基本信息")
print("=" * 60)

print("数据形状：", df.shape)

print("\n列名：")
print(df.columns.tolist())

print("\n数据类型：")
print(df.dtypes)


# =========================
# 4. 日期信息
# =========================

df["Date"] = pd.to_datetime(df["Date"])

print("\n" + "=" * 60)
print("2. 日期信息")
print("=" * 60)

print("开始日期：", df["Date"].min())
print("结束日期：", df["Date"].max())

print("交易日数量：", df["Date"].nunique())


# =========================
# 5. 股票数量
# =========================

print("\n" + "=" * 60)
print("3. 股票数量")
print("=" * 60)

print("股票数量：", df["Ticker"].nunique())


# =========================
# 6. 缺失值检查
# =========================

print("\n" + "=" * 60)
print("4. 缺失值检查")
print("=" * 60)

missing = df.isna().sum()

print(missing)

print("\n缺失率（%）：")

missing_rate = (df.isna().mean() * 100).round(2)

print(missing_rate)


# =========================
# 7. 重复值检查
# =========================

print("\n" + "=" * 60)
print("5. 重复值检查")
print("=" * 60)

duplicate_count = df.duplicated(subset=["Date", "Ticker"]).sum()

print("Date + Ticker 重复记录数：", duplicate_count)


# =========================
# 8. 市场区分
# =========================

print("\n" + "=" * 60)
print("6. 市场区分")
print("=" * 60)

print(df["市場・商品区分"].value_counts())


# =========================
# 9. 行业数量
# =========================

print("\n" + "=" * 60)
print("7. 行业数量")
print("=" * 60)

print("33行业数量：", df["33業種区分"].nunique())

print("\n各行业股票数量：")

industry_counts = (
    df[["Ticker", "33業種区分"]].drop_duplicates()["33業種区分"].value_counts()
)

print(industry_counts)


# =========================
# 10. 规模区分
# =========================

print("\n" + "=" * 60)
print("8. 规模区分")
print("=" * 60)

size_counts = df[["Ticker", "規模区分"]].drop_duplicates()["規模区分"].value_counts()

print(size_counts)


# =========================
# 11. Return 基本统计
# =========================

print("\n" + "=" * 60)
print("9. Daily Return 基本统计")
print("=" * 60)

print(df["Return"].describe())


# =========================
# 12. Volatility 基本统计
# =========================

print("\n" + "=" * 60)
print("10. 20日波动率基本统计")
print("=" * 60)

print(df["Volatility_20D"].describe())


# =========================
# 13. Volume 基本统计
# =========================

print("\n" + "=" * 60)
print("11. 成交量基本统计")
print("=" * 60)

print(df["Volume"].describe())


# =========================
# 14. 每个交易日股票数量
# =========================

print("\n" + "=" * 60)
print("12. 每个交易日的股票数量")
print("=" * 60)

daily_stock_count = df.groupby("Date")["Ticker"].nunique()

print(daily_stock_count.describe())


# =========================
# 15. 每个行业的平均收益率
# =========================

print("\n" + "=" * 60)
print("13. 各行业平均 Daily Return")
print("=" * 60)

industry_return = df.groupby("33業種区分")["Return"].mean().sort_values(ascending=False)

print(industry_return)


# =========================
# 16. 各行业平均波动率
# =========================

print("\n" + "=" * 60)
print("14. 各行业平均 20D Volatility")
print("=" * 60)

industry_volatility = (
    df.groupby("33業種区分")["Volatility_20D"].mean().sort_values(ascending=False)
)

print(industry_volatility)


print("\n" + "=" * 60)
print("EDA 基础检查完成！")
print("=" * 60)
