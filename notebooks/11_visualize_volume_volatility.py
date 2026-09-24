import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "stock_analysis_dataset.csv"

print("正在读取分析数据...")

df = pd.read_csv(input_file)

print("读取完成！")

# ==========================================
# 1. 设置字体
# ==========================================

plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 2. 删除波动率缺失值
# ==========================================

df = df.dropna(subset=["Volatility_20D", "Volume"]).copy()

# ==========================================
# 3. 对成交量进行对数转换
# ==========================================

df["Log_Volume"] = np.log1p(df["Volume"])

# ==========================================
# 4. 随机抽取部分数据用于绘图
# ==========================================

sample = df.sample(n=min(50000, len(df)), random_state=42)

print("\n用于绘图的观察值：", len(sample))

# ==========================================
# 5. 计算相关系数
# ==========================================

correlation = df["Log_Volume"].corr(df["Volatility_20D"])

print("\nLog(Volume) 与 20日波动率的相关系数：")
print(correlation)

# ==========================================
# 6. 绘制散点图
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(sample["Log_Volume"], sample["Volatility_20D"] * 100, alpha=0.15, s=8)

plt.xlabel("Log(Trading Volume + 1)")
plt.ylabel("20-Day Volatility (%)")

plt.title("Trading Activity and Stock Return Volatility")

plt.tight_layout()

# ==========================================
# 7. 保存
# ==========================================

output_file = project_root / "data" / "volume_volatility_scatter.png"

plt.savefig(output_file, dpi=300)

plt.show()

print("\n图表已保存：")
print(output_file)
