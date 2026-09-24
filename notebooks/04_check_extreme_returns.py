import pandas as pd
from pathlib import Path

# =========================
# 1. 项目路径
# =========================

project_root = Path(__file__).resolve().parent.parent

data_file = project_root / "data" / "stock_analysis_dataset.csv"


# =========================
# 2. 读取数据
# =========================

print("正在读取数据...")

df = pd.read_csv(data_file)

df["Date"] = pd.to_datetime(df["Date"])

print("读取完成！")


# =========================
# 3. 极端上涨
# =========================

print("\n" + "=" * 60)
print("1. 极端上涨（Return > 50%）")
print("=" * 60)

large_positive = df[df["Return"] > 0.50][
    ["Date", "Ticker", "銘柄名", "Return", "市場・商品区分", "33業種区分"]
].sort_values("Return", ascending=False)

print("记录数量：", len(large_positive))

print("\n前20条：")

print(large_positive.head(20).to_string(index=False))


# =========================
# 4. 极端下跌
# =========================

print("\n" + "=" * 60)
print("2. 极端下跌（Return < -50%）")
print("=" * 60)

large_negative = df[df["Return"] < -0.50][
    ["Date", "Ticker", "銘柄名", "Return", "市場・商品区分", "33業種区分"]
].sort_values("Return")

print("记录数量：", len(large_negative))

print("\n前20条：")

print(large_negative.head(20).to_string(index=False))


# =========================
# 5. 极端值数量
# =========================

print("\n" + "=" * 60)
print("3. 极端值数量")
print("=" * 60)

print("Return > +50%：", (df["Return"] > 0.50).sum())

print("Return < -50%：", (df["Return"] < -0.50).sum())

print("Return > +100%：", (df["Return"] > 1.00).sum())

print("Return < -90%：", (df["Return"] < -0.90).sum())


# =========================
# 6. Return 分布分位数
# =========================

print("\n" + "=" * 60)
print("4. Return 分位数")
print("=" * 60)

quantiles = df["Return"].quantile(
    [0.001, 0.005, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.995, 0.999]
)

print(quantiles)


print("\n" + "=" * 60)
print("极端 Return 检查完成！")
print("=" * 60)
