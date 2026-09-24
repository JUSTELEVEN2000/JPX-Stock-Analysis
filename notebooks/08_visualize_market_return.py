import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "stock_analysis_dataset.csv"

print("正在读取分析数据...")

df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"])

print("读取完成！")

# ==========================================
# 1. 计算每日等权平均收益率
# ==========================================

market_return = df.groupby("Date")["Return"].mean()

# ==========================================
# 2. 找出最大 / 最小值
# ==========================================

max_date = market_return.idxmax()
max_return = market_return.max()

min_date = market_return.idxmin()
min_return = market_return.min()

print("\n市场每日平均收益率：")
print(market_return.describe())

print("\n最大单日平均收益率：")
print(max_date)
print(max_return)

print("\n最小单日平均收益率：")
print(min_date)
print(min_return)

# ==========================================
# 3. 绘图
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(market_return.index, market_return.values, linewidth=1)

# 0%基准线
plt.axhline(0, linewidth=0.8)

# 标记最大值
plt.scatter(max_date, max_return)

plt.annotate(
    f"{max_date:%Y-%m-%d}\n+{max_return:.2%}",
    xy=(max_date, max_return),
    xytext=(10, -25),
    textcoords="offset points",
)

# 标记最小值
plt.scatter(min_date, min_return)

plt.annotate(
    f"{min_date:%Y-%m-%d}\n{min_return:.2%}",
    xy=(min_date, min_return),
    xytext=(10, 10),
    textcoords="offset points",
)

plt.title("Equal-Weighted Average Daily Stock Return in Japan")

plt.xlabel("Date")
plt.ylabel("Average Daily Return")

plt.tight_layout()

# ==========================================
# 4. 保存
# ==========================================

output_file = project_root / "data" / "market_average_return.png"

plt.savefig(output_file, dpi=300)

plt.show()

print("\n图表已保存：")
print(output_file)
