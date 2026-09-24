import pandas as pd
from pathlib import Path

# =========================
# 1. 项目路径
# =========================

project_root = Path(__file__).resolve().parent.parent

price_file = project_root / "data" / "stock_prices_raw.csv"


# =========================
# 2. 读取价格数据
# =========================

print("正在读取价格数据...")

price_data = pd.read_csv(price_file, header=[0, 1], index_col=0)

print("\n读取完成！")


# =========================
# 3. 基本信息
# =========================

print("\n=========================")
print("数据基本信息")
print("=========================")

print("数据形状：", price_data.shape)

print("\n前5行：")
print(price_data.head())

print("\n列层级：")
print(price_data.columns.names)


# =========================
# 4. 股票数量
# =========================

tickers = price_data.columns.get_level_values(1).unique()

print("\n=========================")
print("股票数量")
print("=========================")

print("股票数量：", len(tickers))


# =========================
# 5. 日期范围
# =========================

print("\n=========================")
print("日期范围")
print("=========================")

print("开始日期：", price_data.index.min())
print("结束日期：", price_data.index.max())


# =========================
# 6. Close 数据
# =========================

close_data = price_data["Close"]

print("\n=========================")
print("Close 数据")
print("=========================")

print("Close 数据形状：", close_data.shape)


# =========================
# 7. 缺失值检查
# =========================

valid_counts = close_data.notna().sum()

print("\n=========================")
print("缺失数据检查")
print("=========================")

print("平均每家公司有效价格数量：", valid_counts.mean())

print("最少有效价格数量：", valid_counts.min())

print("最多有效价格数量：", valid_counts.max())


# =========================
# 8. 完全没有数据的股票
# =========================

missing_tickers = valid_counts[valid_counts == 0].index

print("\n完全没有价格数据的股票数量：", len(missing_tickers))

if len(missing_tickers) > 0:
    print("\n完全没有价格数据的股票：")
    print(missing_tickers.tolist())
