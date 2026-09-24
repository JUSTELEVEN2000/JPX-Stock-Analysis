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

# ==========================================
# 2. 计算各行业平均20日波动率
# ==========================================

industry_volatility = df.groupby("33業種区分")["Volatility_20D"].mean().sort_values()

# ==========================================
# 3. 计算各行业股票数量
# ==========================================

industry_count = df.groupby("33業種区分")["Ticker"].nunique()

# ==========================================
# 4. 合并结果
# ==========================================

result = pd.DataFrame(
    {"Average_Volatility": industry_volatility, "Stock_Count": industry_count}
)

print("\n行业波动率：")
print(result)

# ==========================================
# 5. 绘图
# ==========================================

plt.figure(figsize=(10, 8))

plt.barh(result.index, result["Average_Volatility"] * 100)

plt.xlabel("Average 20-Day Volatility (%)")
plt.ylabel("Industry")

plt.title("Average 20-Day Volatility by Industry")

plt.tight_layout()

# ==========================================
# 6. 保存
# ==========================================

output_file = project_root / "data" / "industry_volatility.png"

plt.savefig(output_file, dpi=300)

plt.show()

print("\n图表已保存：")
print(output_file)
