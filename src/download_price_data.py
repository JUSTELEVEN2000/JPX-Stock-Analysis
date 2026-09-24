import pandas as pd
import yfinance as yf
from pathlib import Path
import time

# =========================
# 1. 项目路径
# =========================

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "jpx_domestic_stocks.csv"
output_file = project_root / "data" / "stock_prices_raw.csv"


# =========================
# 2. 读取 JPX 股票名单
# =========================

stocks = pd.read_csv(input_file)

print("JPX 国内普通股票数量：", len(stocks))


# =========================
# 3. 创建 Yahoo Finance ticker
# =========================

stocks["ticker"] = stocks["コード"].astype(str) + ".T"

tickers = stocks["ticker"].tolist()

print("准备下载股票数量：", len(tickers))


# =========================
# 4. 分批下载
# =========================

batch_size = 100

all_data = []

total_batches = (len(tickers) + batch_size - 1) // batch_size

for i in range(0, len(tickers), batch_size):

    batch = tickers[i : i + batch_size]

    batch_number = i // batch_size + 1

    print("\n" + "=" * 60)
    print(f"正在下载第 {batch_number} / {total_batches} 批")
    print(f"股票范围：{i + 1} - {min(i + batch_size, len(tickers))}")
    print("=" * 60)

    try:

        data = yf.download(
            batch,
            start="2025-01-01",
            end="2026-09-01",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        all_data.append(data)

        print("本批下载完成：", data.shape)

    except Exception as e:

        print("本批下载失败：", e)

    time.sleep(2)


# =========================
# 5. 合并数据
# =========================

print("\n开始合并所有批次数据...")

if all_data:

    price_data = pd.concat(all_data, axis=1)

    print("最终数据形状：", price_data.shape)

    price_data.to_csv(output_file)

    print("\n价格数据已保存：")
    print(output_file)

else:

    print("没有成功下载任何数据。")
