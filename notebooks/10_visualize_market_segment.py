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
# 2. 计算不同市场的收益率统计
# ==========================================

segment_stats = df.groupby("市場・商品区分")["Return"].agg(
    Mean="mean", Median="median", Std="std", Count="count"
)

print("\n不同市场的收益率统计：")
print(segment_stats)

# ==========================================
# 3. 绘制箱线图
# ==========================================

segments = ["プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）"]

data = [
    df.loc[df["市場・商品区分"] == segment, "Return"].dropna() for segment in segments
]

plt.figure(figsize=(10, 6))

plt.boxplot(data, tick_labels=["Prime", "Standard", "Growth"], showfliers=False)

plt.axhline(0, linewidth=0.8)

plt.ylabel("Daily Return")

plt.title("Distribution of Daily Stock Returns by Market Segment")

plt.tight_layout()

# ==========================================
# 4. 保存
# ==========================================

output_file = project_root / "data" / "market_segment_return_distribution.png"

plt.savefig(output_file, dpi=300)

plt.show()

print("\n图表已保存：")
print(output_file)
