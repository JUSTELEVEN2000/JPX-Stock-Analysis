import pandas as pd
from pathlib import Path

# ============================================================
# 0. 路径
# ============================================================

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "stock_analysis_dataset.csv"

output_file = project_root / "data" / "robustness_check.csv"

# ============================================================
# 1. 读取数据
# ============================================================

print("正在读取分析数据...")

df = pd.read_csv(input_file)

print("读取完成！")

# ============================================================
# 2. 原始平均收益率
# ============================================================

original = df.groupby("33業種区分")["Return"].mean().rename("Original_Mean_Return")

# ============================================================
# 3. 去除极端收益率
# ============================================================

# 保留 -50% ~ +50% 范围内的观察值

df_trimmed = df[df["Return"].between(-0.50, 0.50)].copy()

print("\n原始观察值：", len(df))
print("去除极端值后：", len(df_trimmed))
print("去除观察值：", len(df) - len(df_trimmed))

# ============================================================
# 4. 去除极端值后的平均收益率
# ============================================================

trimmed = (
    df_trimmed.groupby("33業種区分")["Return"].mean().rename("Trimmed_Mean_Return")
)

# ============================================================
# 5. 合并
# ============================================================

result = pd.concat([original, trimmed], axis=1)

# 转换为百分比
result["Original_Mean_Return_pct"] = result["Original_Mean_Return"] * 100

result["Trimmed_Mean_Return_pct"] = result["Trimmed_Mean_Return"] * 100

# 差异
result["Difference_pct_point"] = (
    result["Trimmed_Mean_Return_pct"] - result["Original_Mean_Return_pct"]
)

# 按原始平均收益率排序
result = result.sort_values("Original_Mean_Return", ascending=False)

# ============================================================
# 6. 输出
# ============================================================

print("\n========================================")
print("稳健性分析结果")
print("========================================")

print(
    result[
        ["Original_Mean_Return_pct", "Trimmed_Mean_Return_pct", "Difference_pct_point"]
    ].to_string()
)

result.to_csv(output_file, encoding="utf-8-sig")

print("\n结果已保存：")
print(output_file)

# ============================================================
# 7. 计算整体变化
# ============================================================

overall_original = df["Return"].mean()
overall_trimmed = df_trimmed["Return"].mean()

print("\n========================================")
print("整体结果")
print("========================================")

print("原始平均收益率：", round(overall_original * 100, 4), "%")

print("去除极端值后：", round(overall_trimmed * 100, 4), "%")

print("变化：", round((overall_trimmed - overall_original) * 100, 4), "个百分点")

print("\n稳健性分析完成！")
