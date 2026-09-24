import pandas as pd
from pathlib import Path

# 项目根目录
project_root = Path(__file__).resolve().parent.parent

# JPX 原始数据
file_path = project_root / "data" / "data_j.xlsx"

# 读取 Excel
df = pd.read_excel(file_path)

# 查看数据基本情况
print("数据行数：", len(df))
print("数据列数：", len(df.columns))

print("\n列名：")
print(df.columns.tolist())

print("\n前5行：")
print(df.head())

print("\n市场・商品区分：")
print(df["市場・商品区分"].value_counts())

print("\n33業種区分：")
print(df["33業種区分"].value_counts())

print("\n規模区分：")
print(df["規模区分"].value_counts())

# 保留日本国内普通股票
stock_markets = [
    "プライム（内国株式）",
    "スタンダード（内国株式）",
    "グロース（内国株式）",
]

stocks = df[df["市場・商品区分"].isin(stock_markets)].copy()

print("\n日本国内普通股票数量：", len(stocks))

print("\n市场分布：")
print(stocks["市場・商品区分"].value_counts())

print("\n行业数量：")
print(stocks["33業種区分"].value_counts())

# =========================
# 保存日本国内普通股票数据
# =========================

output_file = project_root / "data" / "jpx_domestic_stocks.csv"

stocks.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n已保存文件：")
print(output_file)

print("\n最终数据行数：", len(stocks))

# =========================
# 数据质量检查
# =========================

print("\n=========================")
print("数据质量检查")
print("=========================")

# 1. 检查缺失值
print("\n各列缺失值数量：")
print(stocks.isna().sum())

# 2. 检查股票代码是否重复
print("\n股票代码重复数量：")
print(stocks["コード"].duplicated().sum())

# 3. 检查股票代码数量
print("\n股票代码唯一数量：")
print(stocks["コード"].nunique())

# 4. 检查数据类型
print("\n数据类型：")
print(stocks.dtypes)

# =========================
# 数据类型处理
# =========================

# 将日期转换为 datetime
stocks["日付"] = pd.to_datetime(stocks["日付"].astype(str), format="%Y%m%d")

# 股票代码明确保存为字符串
stocks["コード"] = stocks["コード"].astype(str)

print("\n处理后的数据类型：")
print(stocks.dtypes)

print("\n日期范围：")
print(stocks["日付"].min(), "→", stocks["日付"].max())

print("\n处理后的前5行：")
print(stocks[["日付", "コード", "銘柄名"]].head())
