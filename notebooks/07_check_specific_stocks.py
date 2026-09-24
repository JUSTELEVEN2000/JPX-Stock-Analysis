import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
price_file = project_root / "data" / "stock_prices_raw.csv"

print("正在读取价格数据...")

price_data = pd.read_csv(price_file, header=[0, 1], index_col=0)

price_data.index = pd.to_datetime(price_data.index)

print("读取完成！")


# ==========================================
# 检查指定股票
# ==========================================

tickers = ["5537.T", "8303.T", "7946.T", "8227.T"]


for ticker in tickers:

    print("\n" + "=" * 70)
    print(f"{ticker} 价格检查")
    print("=" * 70)

    if ticker not in price_data["Close"].columns:
        print("没有找到该股票")
        continue

    result = pd.DataFrame(
        {
            "Close": price_data["Close"][ticker],
            "Adj_Close": price_data["Adj Close"][ticker],
            "Volume": price_data["Volume"][ticker],
        }
    )

    result = result.dropna(subset=["Close"])

    result["Return_Close"] = result["Close"].pct_change(fill_method=None)

    result["Return_Adj"] = result["Adj_Close"].pct_change(fill_method=None)

    # 只显示出现大幅变化的日期
    extreme = result[
        (result["Return_Close"].abs() > 0.30) | (result["Return_Adj"].abs() > 0.30)
    ]

    print("\n【大幅价格变化】")
    print(extreme.to_string())

print("\n" + "=" * 70)
print("检查完成")
print("=" * 70)
