import pandas as pd
from pathlib import Path

# =========================
# 1. 路径
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

input_file = DATA_DIR / "stock_feature_dataset.csv"
output_file = DATA_DIR / "tableau_stock_data.csv"

# =========================
# 2. 读取数据
# =========================

print("读取数据...")

df = pd.read_csv(input_file, low_memory=False)

print(f"原始数据: {df.shape}")

# =========================
# 3. Tableau需要的字段
# =========================

tableau_columns = [
    "Date",
    "Ticker",
    "銘柄名",
    "市場・商品区分",
    "33業種区分",
    "17業種区分",
    "規模区分",
    "Return",
    "Return_5D",
    "Return_20D",
    "Volatility_20D",
    "Volume",
    "MovingAverage_20D",
    "Volume_Change_5D",
]

df = df[tableau_columns].copy()

# =========================
# 4. 日期格式
# =========================

df["Date"] = pd.to_datetime(df["Date"])

# =========================
# 5. 保存
# =========================

df.to_csv(output_file, index=False, encoding="utf-8-sig")

print()
print("Tableau数据准备完成！")
print(f"输出文件: {output_file}")
print(f"数据大小: {df.shape}")
print()
print("字段:")
print(df.columns.tolist())
