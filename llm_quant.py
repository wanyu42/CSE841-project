import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import re
import warnings
import requests
import os
import xgboost as xgb
from xgboost import XGBRegressor
from tabulate import tabulate
from dateutil.relativedelta import relativedelta
import argparse
import matplotlib.pyplot as plt
import transformers
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Initialize Transformers pipeline for LLM
LLM_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # Ensure this model is available locally or via Hugging Face
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
generator = transformers.pipeline(
    "text-generation", 
    model=LLM_MODEL_NAME, 
    device=device,
    torch_dtype=torch.bfloat16
)

# Financial Modeling Prep API Key
FMP_API_KEY = 'ZEYdRqMj64hyx91Tr1k9CV5upBY7oxwK'  # Replace with your actual FMP API key

# Directories to store data and results
CACHE_DIR = 'financial_data'
RESULTS_DIR = './results'
PORTFOLIO_DIR = './portfolio'
PLOTS_DIR = './plots'

for directory in [CACHE_DIR, RESULTS_DIR, PORTFOLIO_DIR, PLOTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# List of S&P 500 tickers (top 50 by market cap as an example)
sp500_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META',
    'BRK.B', 'UNH', 'JNJ', 'V', 'XOM', 'PG', 'JPM', 'MA', 'HD',
    'BAC', 'CVX', 'ABBV', 'KO', 'PFE', 'PEP', 'MRK', 'AVGO',
    'CSCO', 'COST', 'TMO', 'DIS', 'ACN', 'WMT', 'ABT', 'DHR',
    'ADBE', 'CRM', 'LIN', 'CMCSA', 'MCD', 'TXN', 'NFLX', 'NEE',
    'PM', 'AMD', 'T', 'LOW', 'HON', 'INTC', 'MDT', 'AMGN', 'SBUX',
]

def get_top_sp500_stocks(n=5):
    """
    Retrieve the top n S&P 500 stocks based on market capitalization.
    """
    market_caps = {}
    for ticker in sp500_tickers:
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get('marketCap', 0)
            if market_cap:
                market_caps[ticker] = market_cap
        except Exception as e:
            print(f"Error retrieving data for {ticker}: {e}")
            continue
    # Sort by market cap and get top n
    top_stocks = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)[:n]
    return [stock[0] for stock in top_stocks]

def ensure_ticker_directory(ticker):
    """
    Ensure that the directory for a given ticker exists.
    """
    ticker_dir = os.path.join(CACHE_DIR, ticker)
    if not os.path.exists(ticker_dir):
        os.makedirs(ticker_dir)
    return ticker_dir

def get_financial_statement_cached(ticker, statement_type='income-statement', period='annual', limit=100):
    """
    Fetch financial statements from local cache or FMP API if not cached.
    
    Parameters:
    - ticker (str): Stock ticker symbol.
    - statement_type (str): 'income-statement', 'balance-sheet-statement', or 'cash-flow-statement'.
    - period (str): 'annual' or 'quarter'.
    - limit (int): Number of records to fetch.
    
    Returns:
    - DataFrame: Financial statement data.
    """
    # Define the file path
    ticker_dir = ensure_ticker_directory(ticker)
    filename = f"{statement_type}.csv"
    file_path = os.path.join(ticker_dir, filename)
    
    # Check if the file exists
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['date'])
        return df
    else:
        # Fetch from FMP API
        url = f'https://financialmodelingprep.com/api/v3/{statement_type}/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                # Save to CSV
                df.to_csv(file_path, index=False)
                print(f"Saved {statement_type} for {ticker} to {file_path}")
                return df
            else:
                print(f"No data found for {ticker} in {statement_type}.")
                return pd.DataFrame()
        else:
            print(f"Error fetching {statement_type} for {ticker}: {response.status_code}")
            return pd.DataFrame()

def get_financial_data_fmp(ticker):
    """
    Fetch all required financial statements for a given ticker from FMP with caching.
    
    Parameters:
    - ticker (str): Stock ticker symbol.
    
    Returns:
    - Tuple[DataFrame, DataFrame, DataFrame]: (income_statements, balance_sheets, cash_flows)
    """
    income_statements = get_financial_statement_cached(ticker, 'income-statement', 'annual', limit=100)
    balance_sheets = get_financial_statement_cached(ticker, 'balance-sheet-statement', 'annual', limit=100)
    cash_flows = get_financial_statement_cached(ticker, 'cash-flow-statement', 'annual', limit=100)
    return income_statements, balance_sheets, cash_flows

def format_financial_statements_for_llm(income_statement, cash_flow, balance_sheet):
    """
    Convert financial statements to string format suitable for LLM.
    """
    formatted_income_statement = income_statement.to_string(index=False)
    formatted_cash_flow = cash_flow.to_string(index=False)
    formatted_balance_sheet = balance_sheet.to_string(index=False)
    return formatted_income_statement, formatted_cash_flow, formatted_balance_sheet

def create_prompt_for_fundamental_analysis(current_fs, previous_fs):
    """
    Create a prompt for the LLM to evaluate financial statements.
    """
    current_income_statement, current_cash_flow, current_balance_sheet = current_fs
    previous_income_statement, previous_cash_flow, previous_balance_sheet = previous_fs
    prompt = f"""Evaluate the following financial statements for the current year and the previous year. Provide a score between 0 and 10 for each criterion, where 0 is very poor and 10 is excellent. Consider criteria from the income statement, cash flow statement, and balance sheet. Additionally, provide an overall score based on the average of the criteria scores.

**Income Statement for the Current Year:**
{current_income_statement}

**Cash Flow Statement for the Current Year:**
{current_cash_flow}

**Balance Sheet for the Current Year:**
{current_balance_sheet}

**Income Statement for the Previous Year:**
{previous_income_statement}

**Cash Flow Statement for the Previous Year:**
{previous_cash_flow}

**Balance Sheet for the Previous Year:**
{previous_balance_sheet}

**Criteria for Evaluation:**

*Income Statement:*
1. **Revenue Growth:** Analyze the growth in revenue compared to the previous year.
2. **Gross Profit Margin:** Calculate as Gross Profit / Total Revenue.
3. **Operating Margin:** Calculate as Operating Income / Total Revenue.
4. **Net Profit Margin:** Calculate as Net Income / Total Revenue.
5. **EPS Growth:** Compare EPS to the previous year.
6. **Operating Efficiency:** Consider Operating Expense relative to Total Revenue.

*Cash Flow Statement:*
7. **Operating Cash Flow Growth:** Analyze the growth in cash from operating activities.
8. **Free Cash Flow:** Evaluate Free Cash Flow and its growth.
9. **Cash Flow Adequacy Ratio:** Calculate as Cash from Operations / Capital Expenditures.

*Balance Sheet:*
10. **Debt-to-Equity Ratio:** Calculate as Total Liabilities / Shareholders' Equity.
11. **Current Ratio:** Calculate as Current Assets / Current Liabilities.
12. **Return on Assets (ROA):** Calculate as Net Income / Total Assets.
13. **Return on Equity (ROE):** Calculate as Net Income / Shareholders' Equity.
14. **Asset Turnover Ratio:** Calculate as Total Revenue / Average Total Assets.

Provide the score for each criterion and an overall score. Include explanations for each score.

At the end, provide the scores for each section and the overall score in the following format, without intermediate calculation:

**Income Statement Score:** [Average Income Statement Score]
**Cash Flow Statement Score:** [Average Cash Flow Score]
**Balance Sheet Score:** [Average Balance Sheet Score]
**Overall Score:** [Average Score]
"""
    return prompt

def extract_scores(text):
    """
    Extract scores from the LLM's response.
    """
    # Initialize the scores
    income_statement_score = None
    cash_flow_score = None
    balance_sheet_score = None
    overall_score = None

    # Split the text into lines
    lines = text.split('\n')
    for line in lines:
        # Remove leading/trailing whitespace and all asterisks
        clean_line = line.strip().replace('*', '').strip()
        
        # Remove any duplicate colons
        clean_line = re.sub(r'[:]+', ':', clean_line)

        # Use regular expressions to match the section scores
        # Pattern matches: [Section Name] Score: [Score]
        match = re.match(r'(Income Statement|Cash Flow(?: Statement)?|Balance Sheet) Score[:\s]*([\d\.]+)', clean_line, re.IGNORECASE)
        if match:
            section = match.group(1).lower()
            score = float(match.group(2))
            if 'income' in section:
                income_statement_score = score
            elif 'cash flow' in section:
                cash_flow_score = score
            elif 'balance sheet' in section:
                balance_sheet_score = score
            continue  # Move to the next line to avoid incorrect matches

        # Handle Overall Score separately because it might have a calculation
        if clean_line.lower().startswith('overall score'):
            # Remove any duplicate colons
            clean_line = re.sub(r'[:]+', ':', clean_line)
            # Try to find score after equals sign
            match = re.search(r'=\s*([\d\.]+)', clean_line)
            if match:
                overall_score = float(match.group(1))
            else:
                # Try to find number after 'Overall Score:'
                match = re.match(r'Overall Score[:\s]*([\d\.]+)', clean_line, re.IGNORECASE)
                if match:
                    overall_score = float(match.group(1))
            continue
    return {
        'income': income_statement_score, 
        'cash': cash_flow_score, 
        'balance': balance_sheet_score, 
        'overall': overall_score
    }

def evaluate_financial_statements_llm(current_fs, previous_fs):
    """
    Use the LLM to evaluate financial statements and return the scores.
    """
    prompt = create_prompt_for_fundamental_analysis(current_fs, previous_fs)
    try:
        # Generate text using the LLM
        response = generator(
            [{"role": "user", "content": prompt}],
            do_sample=True,
            temperature=0.8,
            top_p=1,
            max_new_tokens=2048
        )
        analysis = response[0]["generated_text"][-1]['content']
        scores = extract_scores(analysis)
        return scores
    except Exception as e:
        print(f"Error during LLM evaluation: {e}")
        return None

def evaluate_stock_fmp(ticker, eval_date):
    """
    Evaluate a single stock using fundamental analysis from FMP.
    """
    income_statements, balance_sheets, cash_flows = get_financial_data_fmp(ticker)
    
    # Filter financial statements up to the eval_date
    income_statements = income_statements[pd.to_datetime(income_statements['date']) <= eval_date]
    balance_sheets = balance_sheets[pd.to_datetime(balance_sheets['date']) <= eval_date]
    cash_flows = cash_flows[pd.to_datetime(cash_flows['date']) <= eval_date]
    
    # Check if there are enough records
    min_rows = min(len(income_statements), len(balance_sheets), len(cash_flows))
    if min_rows < 2:
        print(f"Not enough financial data for {ticker} as of {eval_date.date()}")
        return {'income': 5, 'cash': 5, 'balance': 5, 'overall': 5}  # Neutral scores if insufficient data
    
    # Use the last 2 annual statements
    income_statements = income_statements.head(2)
    balance_sheets = balance_sheets.head(2)
    cash_flows = cash_flows.head(2)
    
    # Evaluate the last year against the previous year
    current_fs = format_financial_statements_for_llm(
        income_statements.iloc[0],
        cash_flows.iloc[0],
        balance_sheets.iloc[0]
    )
    previous_fs = format_financial_statements_for_llm(
        income_statements.iloc[1],
        cash_flows.iloc[1],
        balance_sheets.iloc[1]
    )
    scores = evaluate_financial_statements_llm(current_fs, previous_fs)
    if scores is not None:
        return scores
    else:
        return {'income': 5, 'cash': 5, 'balance': 5, 'overall': 5}  # Neutral scores in case of error


def decision_making_via_llm(predicted_score, fundamentals_score):
    """Make decision based on LLM"""
    prompt = f"""You are a portfolio manager. Given the following predictions from numerical models and scores from textual analysis, construct a portfolio that meets the following constraints:

1. Risk tolerance: The portfolio should make ***Sharpe Ration*** as large as possible

2. Turnover rate: The turnover rate of the portfolio should be restricted to 10%.

Input:

Numerical Predictions: {predicted_score}

Textual Analysis Scores: {fundamentals_score}

Output format:

1. Reasoning steps for portfolio construction.

2. Final portfolio weights (ensure they sum to 1 and adhere to constraints).
"""
    try:
        # Generate text using the LLM
        response = generator(
            [{"role": "user", "content": prompt}],
            do_sample=True,
            temperature=0.8,
            top_p=1,
            max_new_tokens=2048
        )
        scores = response[0]["generated_text"][-1]['content']
        
        return scores
    except Exception as e:
        print(f"Error during LLM Decision making: {e}")
        return None


def linear_regression_model(price_data, holding_period=5, ticker=None):
    """
    Train Linear Regression model using the lookback_window data.
    """
    # Ensure price_data is sorted by date
    price_data = price_data.sort_index()
    
    # Compute the target variable y
    price_data = price_data.copy()
    price_data['future_close'] = price_data['Adj Close'].shift(-holding_period)
    price_data['buy_close'] = price_data['Adj Close'].shift(-1)
    price_data['y'] = (price_data['future_close'] - price_data['buy_close']) / price_data['buy_close']
    
    # Drop rows with NaN values resulting from shifting
    price_data.dropna(inplace=True)
    
    # Features: all numerical columns except 'future_close', 'buy_close', 'y'
    numerical_cols = price_data.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [('future_close', ''), ('buy_close', ''), ('y', '')]
    feature_columns = [col for col in numerical_cols if col not in exclude_cols]  
    
    # Create lagged features for the past 'holding_period' days
    lagged_features = []
    for lag in range(holding_period, 0, -1):
        shifted = price_data[feature_columns].shift(lag)
        shifted.columns = [f"{col}_lag{lag}" for col in feature_columns]
        lagged_features.append(shifted)
    
    # Concatenate all lagged features into a single DataFrame
    X = pd.concat(lagged_features, axis=1)
    
    # Align the target variable y with the lagged features
    X = X.loc[price_data.index]
    y = price_data['y']
    
    # Drop any rows with NaN values introduced by shifting
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined.drop('y', axis=1).values
    y = combined['y'].values
    
    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)
    
    return model

def xgboost_model(price_data, holding_period=5, ticker=None):
    """
    Train XGBoost regression model using the lookback_window data.
    """
    # Ensure price_data is sorted by date
    price_data = price_data.sort_index()
    
    # Compute the target variable y
    price_data = price_data.copy()
    price_data['future_close'] = price_data['Adj Close'].shift(-holding_period)
    price_data['buy_close'] = price_data['Adj Close'].shift(-1)
    price_data['y'] = (price_data['future_close'] - price_data['buy_close']) / price_data['buy_close']
    
    # Drop rows with NaN values resulting from shifting
    price_data.dropna(inplace=True)
    
    # Features: all numerical columns except 'future_close', 'buy_close', 'y'
    numerical_cols = price_data.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [('future_close', ''), ('buy_close', ''), ('y', '')]
    feature_columns = [col for col in numerical_cols if col not in exclude_cols]
    
    # Create lagged features for the past 'holding_period' days
    lagged_features = []
    for lag in range(holding_period, 0, -1):
        shifted = price_data[feature_columns].shift(lag)
        shifted.columns = [f"{col}_lag{lag}" for col in feature_columns]
        lagged_features.append(shifted)
    
    # Concatenate all lagged features into a single DataFrame
    X = pd.concat(lagged_features, axis=1)
    
    # Align the target variable y with the lagged features
    X = X.loc[price_data.index]
    y = price_data['y']
    
    # Drop any rows with NaN values introduced by shifting
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined.drop('y', axis=1).values
    y = combined['y'].values
    
    # Initialize and train the XGBoost Regressor
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42)
    model.fit(X, y)
    
    return model

def score_to_predicted_return(score):
    """
    Map a fundamental analysis score (0-10) to a predicted return.
    """
    # Map score (0 to 10) to a return between -r_max and +r_max
    r_max = 0.02  # Maximum expected daily return (e.g., 2%)
    # Normalize score to range [-1, 1]
    normalized_score = (score - 5) / 5
    normalized_score = max(min(normalized_score, 1), -1)
    # Predicted return
    predicted_return = normalized_score * r_max
    return predicted_return

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    """
    Calculate the Sharpe Ratio for a series of returns.
    """
    # Calculate excess returns
    excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
    if excess_returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def generate_backtest_periods(backtest_start_date, backtest_end_date, freq='M'):
    """
    Generate backtest start and end dates based on frequency.
    
    Parameters:
    - backtest_start_date (str): Start date in 'YYYY-MM-DD' format.
    - backtest_end_date (str): End date in 'YYYY-MM-DD' format.
    - freq (str): Frequency ('H', 'Q', 'M', 'W') where 'H' is half-year.
    
    Returns:
    - List[str], List[str]: Lists of start and end dates in 'YYYY-MM-DD' format.
    """
    start = datetime.strptime(backtest_start_date, '%Y-%m-%d')
    end = datetime.strptime(backtest_end_date, '%Y-%m-%d')
    start_dates = []
    end_dates = []

    current_start = start
    while current_start <= end:
        if freq == 'H':  # Half-Year
            current_end = current_start + relativedelta(months=6) - timedelta(days=1)
        elif freq == 'Q':  # Quarterly
            current_end = current_start + relativedelta(months=3) - timedelta(days=1)
        elif freq == 'M':  # Monthly
            current_end = current_start + relativedelta(months=1) - timedelta(days=1)
        elif freq == 'W':  # Weekly
            current_end = current_start + relativedelta(weeks=1) - timedelta(days=1)
        else:
            raise ValueError("Unsupported frequency. Choose from 'H', 'Q', 'M', 'W'.")

        if current_end > end:
            current_end = end

        start_dates.append(current_start.strftime('%Y-%m-%d'))
        end_dates.append(current_end.strftime('%Y-%m-%d'))

        # Move to the next period
        if freq == 'H':
            current_start += relativedelta(months=6)
        elif freq == 'Q':
            current_start += relativedelta(months=3)
        elif freq == 'M':
            current_start += relativedelta(months=1)
        elif freq == 'W':
            current_start += relativedelta(weeks=1)

    # Ensure the first start date matches the input start date and the last end date matches the input end date
    if start_dates and start_dates[0] != backtest_start_date:
        start_dates = [backtest_start_date] + start_dates
    if end_dates and end_dates[-1] != backtest_end_date:
        end_dates[-1] = backtest_end_date

    return start_dates, end_dates

def get_latest_score(fundamental_scores, transaction_date):
    """
    Get the latest fundamental score up to the transaction_date.
    If no score is available, return a neutral score (e.g., 5).
    """
    available_dates = [d for d in fundamental_scores.keys() if d <= transaction_date.date()]
    if not available_dates:
        return 5
    latest_date = max(available_dates)
    if fundamental_scores[latest_date] is not None:
        return fundamental_scores[latest_date]
    else:
        return 5

def backtest_strategy_fmp(models, backtest_data, holding_period, strategy_type='long_only', signal_type='regression_llm', transaction_cost=0.0):
    """
    Backtest the strategy based on the models and backtest data.
    
    Parameters:
    - models (dict): Dictionary of models per ticker.
    - backtest_data (dict): Dictionary of price data per ticker (only from back_start to back_end).
    - holding_period (int): Number of days to hold the position.
    - strategy_type (str): 'long_only', 'long_short', 'greedy_long_only', 'greedy_long_short'.
    - signal_type (str): 'regression_llm', 'regression', 'xgboost', 'xgboost_llm'.
    - transaction_cost (float): Transaction cost per trade as a decimal (e.g., 0.001 for 0.1%).
    
    Returns:
    - DataFrame: Backtest results with dates and returns.
    - float: Sharpe Ratio.
    - float: Total Return.
    """
    # Initialize a list to store portfolio returns
    portfolio_returns = []

    # Assuming all tickers have the same trading dates, use the first ticker's dates
    sample_ticker = list(backtest_data.keys())[0]
    dates = backtest_data[sample_ticker].index

    # Start iterating from holding_period to ensure sufficient data
    for i in range(holding_period, len(dates) - holding_period, holding_period):
        transaction_date = dates[i]
        print(f"Transaction Date: {transaction_date.date()}")

        # Evaluate the stock on transaction_date
        scores = {}
        for ticker in models.keys():
            predicted_return = 0

            # Ensure there is enough data for the holding_period before the transaction date
            if i < holding_period:
                continue  # Not enough data

            # Prepare the fundamental score
            fundamental_scores = models[ticker].get('fundamental_scores', {})
            latest_score = get_latest_score(fundamental_scores, transaction_date)
            fundamental_return = score_to_predicted_return(latest_score)
            predicted_return += fundamental_return

            # Prepare the ML-based predictions
            # Get the data for the holding_period before transaction_date within backtest_data
            recent_data = backtest_data[ticker].iloc[i - holding_period:i]
            if len(recent_data) < holding_period:
                continue  # Not enough data

            # Prepare features
            X_new = recent_data.select_dtypes(include=[np.number]).values.flatten().reshape(1, -1)
            
            # Regression-based predictions
            if signal_type in ['regression', 'regression_llm']:
                lr_model = models[ticker].get('linear_regression')
                if lr_model:
                    lr_pred = lr_model.predict(X_new)[0]
                    predicted_return += lr_pred

            if signal_type in ['xgboost', 'xgboost_llm']:
                xgb_model = models[ticker].get('xgboost')
                if xgb_model:
                    xgb_pred = xgb_model.predict(X_new)[0]
                    predicted_return += xgb_pred

            scores[ticker] = predicted_return

        # Determine portfolio weights based on strategy_type
        weights = {}
        if strategy_type == 'long_only':
            # Assign weights based on scores, normalize to sum to 1
            if scores:
                max_score = max(scores.values())
                if max_score <= 0:
                    weights = {ticker: 0 for ticker in scores.keys()}
                else:
                    # Weight is proportional to score if score > 0, else 0
                    unnormalized_weights = {ticker: score if score > 0 else 0 for ticker, score in scores.items()}
                    total_weights = sum(unnormalized_weights.values())
                    if total_weights > 0:
                        weights = {ticker: weight / total_weights for ticker, weight in unnormalized_weights.items()}
                    else:
                        weights = {ticker: 0 for ticker in scores.keys()}
            else:
                weights = {}
        
        elif strategy_type == 'long_short':
            # Assign weights based on scores, normalize to sum to 0
            if scores:
                score_values = np.array(list(scores.values()))
                score_mean = score_values.mean()
                score_std = score_values.std()
                if score_std == 0:
                    weights = {ticker: 0 for ticker in scores.keys()}
                else:
                    unnormalized_weights = {ticker: (score - score_mean) for ticker, score in scores.items()}
                    total_weights = np.sum([abs(weight) for weight in unnormalized_weights.values()])
                    if total_weights > 0:
                        weights = {ticker: weight / total_weights for ticker, weight in unnormalized_weights.items()}
                    else:
                        weights = {ticker: 0 for ticker in scores.keys()}
            else:
                weights = {}
        
        elif strategy_type == 'greedy_long_only':
            # Calculate past holding_period returns for each ticker
            returns = {}
            for ticker in models.keys():
                try:
                    start_price = backtest_data[ticker].iloc[i - holding_period]['Adj Close']
                    end_price = backtest_data[ticker].iloc[i - 1]['Adj Close']
                    return_pct = (end_price - start_price) / start_price
                    returns[ticker] = return_pct
                except IndexError:
                    returns[ticker] = -np.inf  # Assign worst return if data is missing
                    
            # Select top half stocks with highest returns
            sorted_tickers = sorted(returns.keys(), key=lambda x: returns[x].values, reverse=True)
            top_half = sorted_tickers[:len(sorted_tickers) // 2]

            if top_half:
                equal_weight = 1.0 / len(top_half)
                weights = {ticker: equal_weight for ticker in top_half}
                # Assign 0 to other tickers
                for ticker in models.keys():
                    if ticker not in top_half:
                        weights[ticker] = 0
            else:
                weights = {ticker: 0 for ticker in models.keys()}
        
        elif strategy_type == 'greedy_long_short':
            # Calculate past holding_period returns for each ticker
            returns = {}
            for ticker in models.keys():
                try:
                    start_price = backtest_data[ticker].iloc[i - holding_period]['Adj Close']
                    end_price = backtest_data[ticker].iloc[i - 1]['Adj Close']
                    return_pct = (end_price - start_price) / start_price
                    returns[ticker] = return_pct
                except IndexError:
                    returns[ticker] = -np.inf  # Assign worst return if data is missing

            # Sort tickers based on returns
            sorted_tickers = sorted(returns.keys(), key=lambda x: returns[x].values, reverse=True)
            half = len(sorted_tickers) // 2
            longs = sorted_tickers[:half]
            shorts = sorted_tickers[half:]

            weights = {}
            if longs:
                long_weight = 0.5 / len(longs)
                for ticker in longs:
                    weights[ticker] = long_weight
            if shorts:
                short_weight = -0.5 / len(shorts)
                for ticker in shorts:
                    weights[ticker] = short_weight
            # Assign 0 to any tickers not in longs or shorts (if odd number of tickers)
            for ticker in models.keys():
                if ticker not in longs and ticker not in shorts:
                    weights[ticker] = 0

        elif strategy_type == 'equal_long_only':
            # Assign equal weights to all tickers
            equal_weight = 1.0 / len(models)
            weights = {ticker: equal_weight for ticker in models.keys()}

        else:
            # Unsupported strategy_type
            weights = {ticker: 0 for ticker in models.keys()}
            print(f"Unsupported strategy_type: {strategy_type}. Assigning zero weights.")

        # Calculate portfolio return for the holding period
        holding_returns = []
        for ticker, weight in weights.items():
            if weight == 0:
                continue
            try:
                start_price = backtest_data[ticker].loc[transaction_date]['Adj Close']
                end_date = dates[i + holding_period]
                end_price = backtest_data[ticker].loc[end_date]['Adj Close']
                stock_return = (end_price - start_price) / start_price
                # Apply transaction cost
                stock_return -= transaction_cost
                holding_returns.append(weight * stock_return)
            except (IndexError, KeyError):
                # If end_price is not available
                holding_returns.append(0)

        # Portfolio return is sum of weighted returns
        portfolio_return = np.sum(holding_returns)
        portfolio_returns.append({'Date': transaction_date, 'Return': portfolio_return})
    
    # Convert to DataFrame
    portfolio_df = pd.DataFrame(portfolio_returns)
    portfolio_df.set_index('Date', inplace=True)

    # Calculate cumulative returns
    portfolio_df['Cumulative Return'] = (1 + portfolio_df['Return']).cumprod() - 1

    # Calculate statistics
    sharpe_ratio = calculate_sharpe_ratio(portfolio_df['Return'])
    total_return = portfolio_df['Cumulative Return'].iloc[-1]

    return portfolio_df, sharpe_ratio, total_return

def main():
    parser = argparse.ArgumentParser(description='Backtest the strategy.')
    parser.add_argument('--backtest_start_date', type=str, default='2023-06-30', help='Start date of the backtest (YYYY-MM-DD).')
    parser.add_argument('--backtest_end_date', type=str, default='2024-05-29', help='End date of the backtest (YYYY-MM-DD).')
    parser.add_argument('--freq', type=str, default='H', choices=['H', 'Q', 'M', 'W'], help='Frequency of backtest periods: H (Half-year), Q (Quarterly), M (Monthly), W (Weekly).')
    parser.add_argument('--holding_period', type=int, default=5, help='Number of days to hold the position.')
    parser.add_argument('--lookback_window', type=int, default=180, help='Number of days for training data.')
    parser.add_argument('--strategy_type', type=str, default='long_only', choices=['long_only', 'long_short', 'greedy_long_only', 'greedy_long_short', 'equal_long_only'], help='Type of strategy.')
    parser.add_argument('--signal_type', type=str, default='regression', choices=['regression', 'regression_llm', 'xgboost', 'xgboost_llm'], help='Type of signal.')
    parser.add_argument('--num_stocks', type=int, default=5, help='Number of top stocks to consider.')
    parser.add_argument('--save_results', action='store_true', help='Save backtest results to a CSV file.')
    parser.add_argument('--save_portfolio', action='store_true', help='Save portfolio returns to a CSV file.')
    parser.add_argument('--save_plot', action='store_true', help='Save portfolio plot to a file.')
    parser.add_argument('--transaction_cost', type=float, default=0.0, help='Transaction cost per trade as a decimal (e.g., 0.001 for 0.1%).')
    args = parser.parse_args()
    
    # Assign variables from args
    backtest_start_date = args.backtest_start_date
    backtest_end_date = args.backtest_end_date
    freq = args.freq
    holding_period = args.holding_period
    lookback_window = args.lookback_window
    strategy_type = args.strategy_type
    signal_type = args.signal_type
    num_stocks = args.num_stocks
    save_results = args.save_results
    save_portfolio = args.save_portfolio
    save_plot = args.save_plot
    transaction_cost = args.transaction_cost
    
    # Validate strategy_type and signal_type
    valid_strategy_types = ['long_only', 'long_short', 'greedy_long_only', 'greedy_long_short', 'equal_long_only']
    if strategy_type not in valid_strategy_types:
        raise ValueError(f"Invalid strategy_type: {strategy_type}. Choose from {valid_strategy_types}.")

    valid_signal_types = ['regression', 'regression_llm', 'xgboost', 'xgboost_llm']
    if signal_type not in valid_signal_types:
        raise ValueError(f"Invalid signal_type: {signal_type}. Choose from {valid_signal_types}.")

    if 'greedy' in strategy_type or 'equal' in strategy_type:
        assert 'llm' not in signal_type, "LLM-based signals are not supported for greedy strategies."

    # Step 1: Retrieve Top S&P 500 Stocks
    top_stocks = get_top_sp500_stocks(n=num_stocks)
    print(f"Top {num_stocks} S&P 500 stocks:")
    print(top_stocks)
    pd.DataFrame(top_stocks, columns=['Ticker']).to_csv('top_stocks.csv', index=False)

    # Step 2: Generate backtest periods based on frequency
    backtest_start_dates, backtest_end_dates = generate_backtest_periods(backtest_start_date, backtest_end_date, freq=freq)
    
    # Initialize a list to store backtest results
    backtest_results = []
    # Initialize a master DataFrame for combined portfolio returns
    full_portfolio_df = pd.DataFrame()
    
    for back_start, back_end in zip(backtest_start_dates, backtest_end_dates):
        print(f"\nBacktesting period: {back_start} to {back_end}")
        back_start_date_dt = datetime.strptime(back_start, '%Y-%m-%d')
        back_end_date_dt = datetime.strptime(back_end, '%Y-%m-%d')
    
        # Define the training period
        train_end = back_start_date_dt - timedelta(days=1)
        train_start = train_end - timedelta(days=lookback_window)
    
        # Step 3: Fetch historical price data for training
        training_data = {}
        for ticker in top_stocks:
            data = yf.download(ticker, start=train_start, end=train_end + timedelta(days=1))
            if data.empty:
                print(f"No training data found for {ticker}. Skipping.")
                continue
            training_data[ticker] = data
    
        # Step 4: Train models for each ticker
        models = {}
        for ticker in top_stocks:
            if ticker not in training_data:
                continue
            print(f"\nTraining models for {ticker}")
            models[ticker] = {}
            # Train Linear Regression
            lr_model = linear_regression_model(training_data[ticker], holding_period=holding_period, ticker=ticker)
            models[ticker]['linear_regression'] = lr_model
            # Train XGBoost
            xgb_model = xgboost_model(training_data[ticker], holding_period=holding_period, ticker=ticker)
            models[ticker]['xgboost'] = xgb_model
            # Note: Fundamental analysis will be handled separately for efficiency
    
        # Step 5: Fetch backtest_data only for back_start to back_end
        backtest_data = {}
        for ticker in top_stocks:
            data = yf.download(ticker, start=back_start_date_dt, end=back_end_date_dt + timedelta(days=1))
            if data.empty:
                print(f"No backtest data found for {ticker}. Skipping.")
                continue
            backtest_data[ticker] = data
    
        # Step 6: Prepare fundamental scores for each ticker within the backtest period
        for ticker in models.keys():
            models[ticker]['fundamental_scores'] = {}
            if 'llm' in signal_type:
                # Get all statement dates up to back_end
                income_statements, balance_sheets, cash_flows = get_financial_data_fmp(ticker)
                # Assuming annual statements; adjust if using quarterly
                statement_dates = pd.to_datetime(income_statements['date'])
                # Filter statement dates within the backtest period
                statement_dates = statement_dates[(statement_dates >= train_start) & (statement_dates <= back_end_date_dt)]
                # Evaluate at each statement date
                for date in statement_dates:
                    scores = evaluate_stock_fmp(ticker, date)
                    models[ticker]['fundamental_scores'][date.date()] = scores['overall']
                # If no statement dates within the period, assign a neutral score
                if not statement_dates.empty:
                    print(f"Fundamental scores evaluated for {ticker} on dates: {statement_dates.dt.date.tolist()}")
                else:
                    # Assign a neutral score for the entire period if no new statements
                    models[ticker]['fundamental_scores'] = {}
                    # Optionally, evaluate once at the start of the period
                    eval_date = back_start_date_dt
                    scores = evaluate_stock_fmp(ticker, eval_date)
                    models[ticker]['fundamental_scores'][eval_date.date()] = scores['overall']
                    print(f"No new financial statements for {ticker} in this period. Assigned neutral score.")

        # Step 7: Backtest the strategy
        portfolio_df, sharpe_ratio, total_return = backtest_strategy_fmp(
            models=models,
            backtest_data=backtest_data,
            holding_period=holding_period,
            strategy_type=strategy_type,
            signal_type=signal_type,
            transaction_cost=transaction_cost
        )

        # Append to master portfolio DataFrame
        full_portfolio_df = pd.concat([full_portfolio_df, portfolio_df])
    
        # Collect statistics
        backtest_results.append({
            'Backtest Period': f"{back_start} to {back_end}",
            'Total Return': f"{total_return:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}"
        })
    
    if 'llm' in signal_type:
        model_name = LLM_MODEL_NAME.replace('/', '_')
    else:
        model_name = 'none'
    # Step 8: After all backtest periods, save the combined portfolio returns
    if save_portfolio:
        combined_portfolio_filename = f'combined_portfolio_{strategy_type}_{signal_type}__{model_name}_{num_stocks}stocks.csv'
        combined_portfolio_filepath = os.path.join(PORTFOLIO_DIR, combined_portfolio_filename)
        full_portfolio_df.to_csv(combined_portfolio_filepath)
        print(f"Saved combined portfolio returns to {combined_portfolio_filepath}")
    
    # Step 9: Save the backtest results to a CSV file if requested
    if save_results:
        # Determine model_name based on signal_type
        results_filename = f'backtest_{strategy_type}_{signal_type}_{model_name}_{num_stocks}stocks_results.csv'
        results_filepath = os.path.join(RESULTS_DIR, results_filename)
        pd.DataFrame(backtest_results).to_csv(results_filepath, index=False)
        print(f"Saved backtest results to {results_filepath}")
    
    # Step 10: Plot cumulative returns of the entire backtest period and save the plot
    if save_plot and not full_portfolio_df.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(full_portfolio_df['Cumulative Return'], label='Portfolio Cumulative Return')
        plt.title('Cumulative Returns Over Backtest Period')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        # Define plot filename
        plot_filename = f'cumulative_returns_{strategy_type}_{signal_type}_{num_stocks}stocks.png'
        plot_filepath = os.path.join(PLOTS_DIR, plot_filename)
        plt.savefig(plot_filepath)
        plt.close()
        print(f"Saved cumulative returns plot to {plot_filepath}")

    # Display backtest statistics
    print("\nBacktest Results:")
    print(tabulate(backtest_results, headers="keys", tablefmt="grid"))

if __name__ == '__main__':
    main()
