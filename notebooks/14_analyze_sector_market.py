import pandas as pd
import matplotlib.pyplot as plt

print("正在读取分析数据...")

df = pd.read_csv("data/stock_analysis_dataset.csv", parse_dates=["Date"])

print("读取完成！")

# ============================================================
# 1. 行业 × 市场：平均收益率
# ============================================================

print("\n正在计算行业 × 市场收益率...")

sector_market_return = (
    df.groupby(["市場・商品区分", "33業種区分"])["Return"]
    .agg(["mean", "median", "std", "count"])
    .reset_index()
)

sector_market_return.columns = [
    "Market",
    "Sector",
    "Mean_Return",
    "Median_Return",
    "Std_Return",
    "Count",
]

print("\n行业 × 市场收益率：")
print(sector_market_return.head(20))


# ============================================================
# 2. 保存结果
# ============================================================

output_path = "data/sector_market_return.csv"

sector_market_return.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\n分析结果已保存：")
print(output_path)


# ============================================================
# 3. 各市场中收益率最高的行业
# ============================================================

print("\n" + "=" * 60)
print("各市场行业收益率统计")
print("=" * 60)

for market in sector_market_return["Market"].unique():

    temp = sector_market_return[sector_market_return["Market"] == market].sort_values(
        "Mean_Return", ascending=False
    )

    print("\n----------------------------------------")
    print(market)
    print("----------------------------------------")

    print(temp[["Sector", "Mean_Return", "Std_Return", "Count"]].to_string(index=False))


# ============================================================
# 4. 行业 × 市场可视化
# ============================================================

pivot_return = sector_market_return.pivot(
    index="Sector", columns="Market", values="Mean_Return"
)

# 按行业整体平均收益率排序
pivot_return["Overall"] = pivot_return.mean(axis=1)

pivot_return = pivot_return.sort_values("Overall", ascending=False)

pivot_return = pivot_return.drop(columns="Overall")

# 转换成百分比
plot_data = pivot_return * 100

plt.figure(figsize=(12, 12))

plot_data.plot(kind="barh", figsize=(12, 12))

plt.xlabel("Mean Daily Return (%)")
plt.ylabel("Sector")
plt.title("Mean Daily Return by Sector and Market Segment")

plt.axvline(x=0, linewidth=0.8)

plt.tight_layout()

output_fig = "data/sector_market_return_comparison.png"

plt.savefig(output_fig, dpi=300, bbox_inches="tight")

plt.close()

print("\n图表已保存：")
print(output_fig)

print("\n行业 × 市场收益率分析完成！")
