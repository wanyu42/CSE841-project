import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
file_paths = [
    "./portfolio/combined_portfolio_equal_long_only_xgboost__none_10stocks.csv",
    "./portfolio/combined_portfolio_greedy_long_only_xgboost__none_10stocks.csv",
    "./portfolio/combined_portfolio_long_only_regression__none_10stocks.csv",
    "./portfolio/combined_portfolio_long_only_regression_llm__meta-llama_Llama-3.2-1B-Instruct_10stocks.csv",
    "./portfolio/combined_portfolio_long_only_xgboost__none_10stocks.csv",
    "./portfolio/combined_portfolio_long_only_xgboost_llm__meta-llama_Llama-3.2-1B-Instruct_10stocks.csv"
]

# Define labels for each signal
labels = [
    "Equal",
    "Greedy",
    "Regression",
    "Regression with LLM",
    "XGBoost",
    "XGBoost with LLM"
]

# Date range for filtering
start_date = "2023-07-01"
end_date = "2023-12-31"

# Prepare to store cumulative returns
cumulative_returns = {}

# Column names
date_column = "Date"
return_column = "Return"

# Process each file
for file_path, label in zip(file_paths, labels):
    # Read CSV
    df = pd.read_csv(file_path)
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    # Filter by date range
    df_filtered = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
    # Compute cumulative return
    df_filtered['cumulative_return'] = (1 + df_filtered[return_column]).cumprod()
    # Store cumulative return with corresponding label
    cumulative_returns[label] = df_filtered.set_index(date_column)['cumulative_return']

# Plot cumulative returns
plt.figure(figsize=(12, 6))
for label, cumulative_return in cumulative_returns.items():
    plt.plot(cumulative_return, label=label)
plt.title("Cumulative Returns (Long Only) from 2023-07-01 to 2023-12-31")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend(loc="upper left")
plt.grid()
plt.tight_layout()
plt.savefig('pres_return.png')
