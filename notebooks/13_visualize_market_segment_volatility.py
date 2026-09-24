import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "stock_analysis_dataset.csv"

print("正在读取分析数据...")

df = pd.read_csv(input_file)

print("读取完成！")

# ==========================================
# 1. 设置日文字体
# ==========================================

plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 2. 删除波动率缺失值
# ==========================================

df = df.dropna(subset=["Volatility_20D"]).copy()

# ==========================================
# 3. 市场顺序
# ==========================================

segments = ["プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）"]

labels = ["Prime", "Standard", "Growth"]

# ==========================================
# 4. 计算各市场波动率统计
# ==========================================

segment_stats = df.groupby("市場・商品区分")["Volatility_20D"].agg(
    Mean="mean", Median="median", Std="std", Count="count"
)

print("\n不同市场的20日波动率统计：")
print(segment_stats.loc[segments])

# ==========================================
# 5. 准备箱线图数据
# ==========================================

data = [
    df.loc[df["市場・商品区分"] == segment, "Volatility_20D"].dropna() * 100
    for segment in segments
]

# ==========================================
# 6. 绘制箱线图
# ==========================================

plt.figure(figsize=(10, 6))

plt.boxplot(data, tick_labels=labels, showfliers=False)

plt.ylabel("20-Day Volatility (%)")

plt.title("Distribution of 20-Day Volatility by Market Segment")

plt.tight_layout()

# ==========================================
# 7. 保存
# ==========================================

output_file = project_root / "data" / "market_segment_volatility_distribution.png"

plt.savefig(output_file, dpi=300)

plt.show()

print("\n图表已保存：")
print(output_file)
