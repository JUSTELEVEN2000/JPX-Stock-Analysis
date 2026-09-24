import pandas as pd
from pathlib import Path

# ============================================================
# 0. 路径
# ============================================================

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "stock_model_dataset.csv"

# ============================================================
# 1. 读取
# ============================================================

print("正在读取模型数据...")

df = pd.read_csv(input_file, parse_dates=["Date"])

print("读取完成！")

# ============================================================
# 2. 选择一只股票进行验证
# ============================================================

ticker = "1301.T"

stock = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)

# 选择第一个可以计算未来5日收益率的观察值
row_index = 0

row = stock.iloc[row_index]

print("\n========================================")
print("验证股票：", ticker)
print("========================================")

print("\n基准日期：")
print(row["Date"])

print("\n当天 Return：")
print(row["Return"])

print("\n模型中的 Future_5D_Return：")
print(row["Future_5D_Return"])

# ============================================================
# 3. 手动取得未来5个交易日
# ============================================================

future = stock.iloc[row_index + 1 : row_index + 6]

print("\n未来5个交易日：")
print(future[["Date", "Return"]].to_string(index=False))

# ============================================================
# 4. 手动计算复合收益率
# ============================================================

manual_future_return = (1 + future["Return"]).prod() - 1

print("\n========================================")
print("手动计算结果")
print("========================================")

print("手动计算 Future 5D Return：", manual_future_return)

print("模型计算 Future 5D Return：", row["Future_5D_Return"])

difference = manual_future_return - row["Future_5D_Return"]

print("两者差异：", difference)

# ============================================================
# 5. 判断
# ============================================================

if abs(difference) < 1e-10:
    print("\n✓ 验证通过：Future_5D_Return 计算正确！")
else:
    print("\n⚠ 两者存在差异，请检查计算逻辑。")
