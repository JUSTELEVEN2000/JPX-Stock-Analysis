import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "stock_analysis_dataset.csv"

output_file = project_root / "data" / "eda_summary.csv"

print("正在读取分析数据...")

df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"])

print("读取完成！")

# ==========================================
# 1. 基本数据规模
# ==========================================

n_observations = len(df)
n_stocks = df["Ticker"].nunique()
n_trading_days = df["Date"].nunique()

# ==========================================
# 2. 收益率统计
# ==========================================

mean_return = df["Return"].mean()
median_return = df["Return"].median()
std_return = df["Return"].std()

# ==========================================
# 3. 20日波动率
# ==========================================

mean_volatility = df["Volatility_20D"].mean()
median_volatility = df["Volatility_20D"].median()

# ==========================================
# 4. 全市场每日等权平均收益率
# ==========================================

market_return = df.groupby("Date")["Return"].mean()

max_market_return = market_return.max()
min_market_return = market_return.min()

max_market_return_date = market_return.idxmax()
min_market_return_date = market_return.idxmin()

# ==========================================
# 5. 极端收益率
# ==========================================

positive_extreme = (df["Return"] > 0.50).sum()

negative_extreme = (df["Return"] < -0.50).sum()

total_extreme = positive_extreme + negative_extreme

# ==========================================
# 6. Volume × Volatility
# ==========================================

volume_volatility_df = df.dropna(subset=["Volume", "Volatility_20D"]).copy()

volume_volatility_df["Log_Volume"] = np.log1p(volume_volatility_df["Volume"])

volume_volatility_correlation = volume_volatility_df["Log_Volume"].corr(
    volume_volatility_df["Volatility_20D"]
)

# ==========================================
# 7. 汇总
# ==========================================

summary = pd.DataFrame(
    {
        "Metric": [
            "Number of stocks",
            "Number of trading days",
            "Number of observations",
            "Mean daily return",
            "Median daily return",
            "Daily return standard deviation",
            "Mean 20-day volatility",
            "Median 20-day volatility",
            "Maximum equal-weighted market return",
            "Maximum return date",
            "Minimum equal-weighted market return",
            "Minimum return date",
            "Returns above +50%",
            "Returns below -50%",
            "Total extreme returns",
            "Log(volume)-volatility correlation",
        ],
        "Value": [
            n_stocks,
            n_trading_days,
            n_observations,
            mean_return,
            median_return,
            std_return,
            mean_volatility,
            median_volatility,
            max_market_return,
            max_market_return_date.strftime("%Y-%m-%d"),
            min_market_return,
            min_market_return_date.strftime("%Y-%m-%d"),
            positive_extreme,
            negative_extreme,
            total_extreme,
            volume_volatility_correlation,
        ],
    }
)

# ==========================================
# 8. 输出结果
# ==========================================

print("\n==============================")
print("EDA SUMMARY")
print("==============================")

print(summary.to_string(index=False))

summary.to_csv(output_file, index=False)

print("\nSummary 已保存：")
print(output_file)
