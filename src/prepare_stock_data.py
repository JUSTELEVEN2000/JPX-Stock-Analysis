import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "stock_prices_raw.csv"

print("正在读取原始价格数据...")

price_data = pd.read_csv(input_file, header=[0, 1], index_col=0)

print("读取完成！")
print("原始数据形状：", price_data.shape)

# ==========================================
# 1. 提取 Adjusted Close 和 Volume
# ==========================================

adj_close = price_data["Adj Close"].copy()
volume_data = price_data["Volume"].copy()

adj_close.index = pd.to_datetime(adj_close.index)
volume_data.index = pd.to_datetime(volume_data.index)

# ==========================================
# 2. 计算每日收益率
# ==========================================

# 不进行缺失值自动填充
daily_return = adj_close.pct_change(fill_method=None)

# ==========================================
# 3. 计算20日波动率
# ==========================================

volatility_20d = daily_return.rolling(20).std()

# ==========================================
# 4. 计算20日移动平均
# ==========================================

moving_average_20d = adj_close.rolling(20).mean()

# ==========================================
# 5. 极端收益率标记
# ==========================================

extreme_return_flag = (daily_return.abs() > 0.50).astype("int8")

# ==========================================
# 6. 输出数据
# ==========================================

daily_return.to_csv(project_root / "data" / "daily_return.csv")

volatility_20d.to_csv(project_root / "data" / "volatility_20d.csv")

moving_average_20d.to_csv(project_root / "data" / "moving_average_20d.csv")

volume_data.to_csv(project_root / "data" / "volume.csv")

extreme_return_flag.to_csv(project_root / "data" / "extreme_return_flag.csv")

# ==========================================
# 7. 输出检查结果
# ==========================================

print("\n=========================")
print("数据处理完成")
print("=========================")

print("Adjusted Close：", adj_close.shape)
print("Daily Return：", daily_return.shape)
print("20D Volatility：", volatility_20d.shape)
print("20D Moving Average：", moving_average_20d.shape)
print("Volume：", volume_data.shape)

print("\n极端收益率观察值：")
print("Return > +50%：", (daily_return > 0.50).sum().sum())

print("Return < -50%：", (daily_return < -0.50).sum().sum())

print("极端收益率总数：", extreme_return_flag.sum().sum())

print("\n文件已保存到 data/ 文件夹。")
