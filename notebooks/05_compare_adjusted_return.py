import pandas as pd
from pathlib import Path

# =========================
# 1. 项目路径
# =========================

project_root = Path(__file__).resolve().parent.parent

price_file = project_root / "data" / "stock_prices_raw.csv"


# =========================
# 2. 读取原始价格
# =========================

print("正在读取原始价格数据...")

price_data = pd.read_csv(price_file, header=[0, 1], index_col=0)

print("读取完成！")


# =========================
# 3. 提取 Close 和 Adj Close
# =========================

close = price_data["Close"].copy()
adj_close = price_data["Adj Close"].copy()

close.index = pd.to_datetime(close.index)
adj_close.index = pd.to_datetime(adj_close.index)


# =========================
# 4. 计算两种 Return
# =========================

print("\n正在计算两种收益率...")

close_return = close.pct_change()
adj_return = adj_close.pct_change()


# =========================
# 5. 转换成长表
# =========================

close_long = close_return.stack().reset_index()

close_long.columns = ["Date", "Ticker", "Close_Return"]


adj_long = adj_return.stack().reset_index()

adj_long.columns = ["Date", "Ticker", "Adj_Return"]


comparison = close_long.merge(adj_long, on=["Date", "Ticker"], how="inner")


# =========================
# 6. Return 差异
# =========================

comparison["Difference"] = comparison["Close_Return"] - comparison["Adj_Return"]


# =========================
# 7. 极端 Close Return
# =========================

print("\n" + "=" * 60)
print("1. Close Return 极端值")
print("=" * 60)

print("Close Return > +50%：", (comparison["Close_Return"] > 0.50).sum())

print("Close Return < -50%：", (comparison["Close_Return"] < -0.50).sum())


# =========================
# 8. 极端 Adjusted Return
# =========================

print("\n" + "=" * 60)
print("2. Adjusted Return 极端值")
print("=" * 60)

print("Adjusted Return > +50%：", (comparison["Adj_Return"] > 0.50).sum())

print("Adjusted Return < -50%：", (comparison["Adj_Return"] < -0.50).sum())


# =========================
# 9. 最大上涨
# =========================

print("\n" + "=" * 60)
print("3. Adjusted Return 最大上涨")
print("=" * 60)

print(
    comparison.sort_values("Adj_Return", ascending=False)
    .head(20)
    .to_string(index=False)
)


# =========================
# 10. 最大下跌
# =========================

print("\n" + "=" * 60)
print("4. Adjusted Return 最大下跌")
print("=" * 60)

print(comparison.sort_values("Adj_Return").head(20).to_string(index=False))


# =========================
# 11. 差异最大的记录
# =========================

print("\n" + "=" * 60)
print("5. Close 与 Adjusted Return 差异最大的记录")
print("=" * 60)

print(
    comparison.assign(Abs_Difference=comparison["Difference"].abs())
    .sort_values("Abs_Difference", ascending=False)
    .head(20)
    .to_string(index=False)
)


print("\n" + "=" * 60)
print("Adjusted Return 比较完成！")
print("=" * 60)
