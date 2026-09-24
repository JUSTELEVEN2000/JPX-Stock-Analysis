import pandas as pd
from pathlib import Path

# =========================
# 1. 项目路径
# =========================

project_root = Path(__file__).resolve().parent.parent

return_file = project_root / "data" / "daily_return.csv"
volatility_file = project_root / "data" / "volatility_20d.csv"
moving_average_file = project_root / "data" / "moving_average_20d.csv"
volume_file = project_root / "data" / "volume.csv"

stock_file = project_root / "data" / "jpx_domestic_stocks.csv"

output_file = project_root / "data" / "stock_analysis_dataset.csv"


# =========================
# 2. 读取数据
# =========================

print("正在读取数据...")

daily_return = pd.read_csv(return_file, index_col=0)

volatility = pd.read_csv(volatility_file, index_col=0)

moving_average = pd.read_csv(moving_average_file, index_col=0)

volume = pd.read_csv(volume_file, index_col=0)

stocks = pd.read_csv(stock_file)

print("读取完成！")


# =========================
# 3. 转换日期
# =========================

daily_return.index = pd.to_datetime(daily_return.index)
volatility.index = pd.to_datetime(volatility.index)
moving_average.index = pd.to_datetime(moving_average.index)
volume.index = pd.to_datetime(volume.index)


# =========================
# 4. 宽表 → 长表
# =========================

print("\n正在转换数据格式...")

returns_long = daily_return.stack().reset_index()

returns_long.columns = ["Date", "Ticker", "Return"]


volatility_long = volatility.stack().reset_index()

volatility_long.columns = ["Date", "Ticker", "Volatility_20D"]


moving_average_long = moving_average.stack().reset_index()

moving_average_long.columns = ["Date", "Ticker", "MovingAverage_20D"]


volume_long = volume.stack().reset_index()

volume_long.columns = ["Date", "Ticker", "Volume"]


# =========================
# 5. 合并金融指标
# =========================

print("正在合并指标...")

analysis_data = returns_long.merge(volatility_long, on=["Date", "Ticker"], how="left")

analysis_data = analysis_data.merge(
    moving_average_long, on=["Date", "Ticker"], how="left"
)

analysis_data = analysis_data.merge(volume_long, on=["Date", "Ticker"], how="left")


# =========================
# 6. 加入 JPX 股票基本信息
# =========================

print("正在加入股票基本信息...")

stock_info = stocks[
    ["コード", "銘柄名", "市場・商品区分", "33業種区分", "17業種区分", "規模区分"]
].copy()

stock_info["Ticker"] = stock_info["コード"].astype(str) + ".T"

stock_info = stock_info.drop(columns=["コード"])

analysis_data = analysis_data.merge(stock_info, on="Ticker", how="left")


# =========================
# 7. 删除没有收益率的数据
# =========================

analysis_data = analysis_data.dropna(subset=["Return"])


# =========================
# 8. 排序
# =========================

analysis_data = analysis_data.sort_values(["Date", "Ticker"])


# =========================
# 9. 保存
# =========================

print("\n正在保存数据...")

analysis_data.to_csv(output_file, index=False, encoding="utf-8-sig")


# =========================
# 10. 最终检查
# =========================

print("\n=========================")
print("分析数据集创建完成")
print("=========================")

print("数据行数：", len(analysis_data))
print("数据列数：", len(analysis_data.columns))

print("\n列名：")
print(analysis_data.columns.tolist())

print("\n前5行：")
print(analysis_data.head())

print("\n股票数量：")
print(analysis_data["Ticker"].nunique())

print("\n日期范围：")
print(analysis_data["Date"].min(), "→", analysis_data["Date"].max())

print("\n已保存：")
print(output_file)
