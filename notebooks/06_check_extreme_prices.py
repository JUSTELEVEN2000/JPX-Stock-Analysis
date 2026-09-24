import pandas as pd
from pathlib import Path

# =========================
# 1. 项目路径
# =========================

project_root = Path(__file__).resolve().parent.parent

price_file = project_root / "data" / "stock_prices_raw.csv"


# =========================
# 2. 读取 Adjusted Close
# =========================

print("正在读取价格数据...")

price_data = pd.read_csv(price_file, header=[0, 1], index_col=0)

adj_close = price_data["Adj Close"].copy()

adj_close.index = pd.to_datetime(adj_close.index)

print("读取完成！")


# =========================
# 3. 指定极端股票
# =========================

tickers = [
    "5537.T",
    "7603.T",
    "5721.T",
    "6072.T",
    "8303.T",
    "7946.T",
    "8227.T",
    "3134.T",
    "4316.T",
]


# =========================
# 4. 查看前后价格
# =========================

for ticker in tickers:

    print("\n" + "=" * 60)
    print(ticker)
    print("=" * 60)

    if ticker not in adj_close.columns:
        print("没有找到价格数据")
        continue

    series = adj_close[ticker].dropna()

    returns = series.pct_change()

    extreme_dates = returns[(returns > 0.50) | (returns < -0.50)].index

    for date in extreme_dates:

        position = series.index.get_loc(date)

        start = max(0, position - 2)

        end = min(len(series), position + 3)

        print(
            pd.DataFrame(
                {"Adj_Close": series.iloc[start:end], "Return": returns.iloc[start:end]}
            )
        )


print("\n" + "=" * 60)
print("极端价格检查完成！")
print("=" * 60)
