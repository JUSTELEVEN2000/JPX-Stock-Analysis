import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# =========================
# 1. Paths
# =========================
project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "daily_ic.csv"

output_plot = project_root / "data" / "daily_ic_timeseries.png"

# =========================
# 2. Load data
# =========================
df = pd.read_csv(input_file, parse_dates=["Date"])

ic = df["IC"].dropna()

n = len(ic)

# =========================
# 3. Basic statistics
# =========================
mean_ic = ic.mean()
std_ic = ic.std(ddof=1)

standard_error = std_ic / np.sqrt(n)

# =========================
# 4. Simple t-test
# =========================
t_stat = mean_ic / standard_error

p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

# =========================
# 5. 95% confidence interval
# =========================
critical_value = stats.t.ppf(0.975, df=n - 1)

ci_low = mean_ic - critical_value * standard_error

ci_high = mean_ic + critical_value * standard_error

# =========================
# 6. Autocorrelation
# =========================
acf_1 = ic.autocorr(lag=1)
acf_5 = ic.autocorr(lag=5)

# =========================
# 7. Positive / negative ratio
# =========================
positive_ratio = (ic > 0).mean()

negative_ratio = (ic < 0).mean()

# =========================
# 8. Print results
# =========================
print("=" * 60)
print("IC Robustness Analysis")
print("=" * 60)

print(f"Number of days:        {n}")

print(f"Mean IC:               {mean_ic:.6f}")
print(f"IC Std:                {std_ic:.6f}")
print(f"Standard Error:        {standard_error:.6f}")

print(f"t-statistic:           {t_stat:.6f}")
print(f"Two-sided p-value:     {p_value:.6f}")

print(f"95% CI:                " f"[{ci_low:.6f}, {ci_high:.6f}]")

print("\nAutocorrelation:")
print(f"Lag 1:                 {acf_1:.6f}")
print(f"Lag 5:                 {acf_5:.6f}")

print("\nIC Direction:")
print(f"Positive IC days:      {positive_ratio:.2%}")
print(f"Negative IC days:      {negative_ratio:.2%}")

# =========================
# 9. Time-series plot
# =========================
plt.figure(figsize=(12, 6))

plt.plot(df["Date"], df["IC"])

plt.axhline(0, linestyle="--")

plt.axhline(mean_ic, linestyle="--")

plt.xlabel("Date")

plt.ylabel("Daily Cross-sectional IC")

plt.title("Daily Cross-sectional IC Over Time")

plt.tight_layout()

plt.savefig(output_plot, dpi=200)

plt.close()

print("\nSaved:")
print(output_plot)
