# JPX Stock Analysis

A data analysis project on Japanese listed stocks, covering market data collection, exploratory data analysis, feature engineering, machine learning, model diagnostics, and interactive Tableau visualization.

## Overview

This project analyzes daily stock data for Japanese listed companies to explore:

* Cross-sectional stock return patterns
* Volatility differences across market segments and industries
* Trading volume and volatility relationships
* Short-term return prediction using historical market features
* Model performance and robustness
* Interactive stock-level exploration using Tableau

The project is designed as an end-to-end data analysis workflow, from raw market data to visualization and model evaluation.

## Data

### Stock Master Data

Stock master information was obtained from the Tokyo Stock Exchange / JPX listed-company data.

The dataset includes:
* Stock code
* Company name
* Market segment
* 33-industry classification
* 17-industry classification
* Size classification

The analysis focuses on domestic common stocks listed on:
* Prime Market
* Standard Market
* Growth Market

The JPX master-data snapshot used in this project is dated August 31, 2026.

### Price Data

Daily stock price data was collected using `yfinance`.

* **Period:** January 2025 – August 2026
* **Coverage:** ~3,700 stocks | 403 trading days | 1.47 million stock-day observations
* **Method:** Adjusted closing prices are used for return calculations.

## Analysis Pipeline

```text
JPX Listed Company Data
          │
          ▼
Stock Universe Construction
          │
          ▼
Daily Price Data Collection
          │
          ▼
Return / Volatility / Volume Features
          │
          ▼
Exploratory Data Analysis
          │
          ▼
Robustness Checks
          │
          ▼
Future Return Construction
          │
          ▼
Feature Engineering
          │
          ▼
Baseline & Machine Learning Models
          │
          ▼
Model Diagnostics
          │
          ▼
Tableau Interactive Dashboard
```

## Exploratory Data Analysis

The EDA examines market-wide and cross-sectional patterns in Japanese equities.

* Market Return: An equal-weighted cross-sectional average of individual stock returns is used to describe daily market-wide movements. The largest positive and negative market-wide movements in the sample occurred on consecutive trading days in April 2025.

* Market Segment Comparison: Daily returns and 20-day volatility are compared across Prime, Standard, and Growth segments. The Growth segment shows higher average volatility in the sample, while Prime stocks exhibit lower average volatility. These results are descriptive and should not be interpreted as causal relationships.

* Industry Analysis: Stock returns and volatility are aggregated by the JPX 33-industry classification to examine cross-industry differences.

* Trading Volume and Volatility: A positive correlation is observed between log trading volume and 20-day volatility in the sample. This analysis is descriptive and does not establish causality.

## Robustness Checks

Large daily returns were investigated rather than automatically removed. Some extreme observations were associated with:
* Listing transitions
* Stock splits
* Changes in listing status
* Other potential data or market-structure effects

A robustness check excluding observations with absolute daily returns above 50% produced only a small change in the overall mean return. Therefore, extreme observations were retained in the raw analytical dataset while robustness results were reported separately.

## Feature Engineering

The prediction task uses information available up to the end of each trading day.

* Main Features: 1-day return, 5-day cumulative return, 20-day cumulative return, 20-day volatility, 5-day / 20-day / 60-day volatility, 5-day and 20-day volume changes, 20-day and 60-day historical return, Market segment, Industry classification, Company size classification.
* Target: Future 5-trading-day cumulative return.
* Validation: A dedicated validation step was used to ensure that the future-return calculation does not include the current day’s return.

## Machine Learning

Several models were evaluated using a time-based train/test split.

## Models

* Mean-return baseline
* Ridge Regression
* Gradient Boosting

## Evaluation & Diagnostics

* Metrics: MAE, RMSE, R^2
*Diagnostics: Actual vs. predicted returns, Prediction distribution, Residual distribution, Group-level performance, Daily Spearman Information Coefficient (IC)

## Results

* The machine-learning models did not produce a meaningful improvement over the simple baseline for absolute 5-day return prediction.
* The Gradient Boosting model produced predictions that were strongly concentrated around the mean, indicating limited predictive dispersion.
* A daily IC analysis showed a small positive average IC, but the result was not sufficiently stable to support a strong conclusion about predictive power.
* This result is treated as an analytical finding rather than as evidence of a reliable trading strategy.

## Tableau Dashboard

The project includes an interactive Tableau Public dashboard with two main views:

1. Market Overview:
* Daily market average return
* Market segment return comparison
* Industry return comparison
* Industry volatility comparison
* Trading volume vs. volatility

2. Stock Explorer:
* Allows users to select an individual stock and examine: Daily return trend, 20-day volatility trend, Key stock indicators, Historical date range.

🔗 **Tableau Public Dashboard:** [JPX Stock Analysis Market Overview](https://public.tableau.com/app/profile/luyi.shou/viz/JPX_Stock_Analysis/MarketOverview)

## Repository Structure

```text
JPX-Stock-Analysis/
│
├── data/
│   ├── data_j.xlsx
│   ├── jpx_domestic_stocks.csv
│   ├── analysis result CSV files
│   └── visualization PNG files
│
├── notebooks/
│   ├── 01_explore_data.py
│   ├── 02_test_price_data.py
│   ├── ...
│   └── 34_prepare_tableau_data.py
│
├── src/
│   ├── download_price_data.py
│   ├── prepare_stock_data.py
│   └── create_analysis_dataset.py
│
└── .gitignore
```

Note: Large raw and intermediate datasets are excluded from Git tracking via ⁠.gitignore⁠.

## Main Tools

* Languages: Python
* Libraries: pandas, NumPy, scikit-learn, SciPy, matplotlib, yfinance, openpyxl
* Visualization & Tooling: Tableau Public, Git / GitHub

## Key Takeaways

This project demonstrates an end-to-end workflow for financial market data analysis:
1. Constructing a stock universe from structured JPX data
2. Collecting and processing large-scale daily price data
3. Performing cross-sectional and time-series EDA
4. Investigating abnormal observations and robustness
5. Constructing leakage-aware prediction features
6. Evaluating machine-learning models against a simple baseline
7. Diagnosing model limitations rather than focusing only on headline metrics
8. Building an interactive Tableau dashboard
9. Managing the project using Git and GitHub

## Disclaimer

This project is for educational and analytical purposes only. The results do not constitute investment advice or a recommendation to buy or sell any security.
