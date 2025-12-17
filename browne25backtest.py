import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
import datetime

# Define start and end dates
start_date = '2004-12-01'  # Start slightly before to ensure Jan 2005 data
end_date = datetime.date.today().strftime('%Y-%m-%d')

# Download monthly data with error handling
def fetch_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, interval='1mo', auto_adjust=False)['Adj Close']
        if data.empty or len(data) < 2:
            raise ValueError(f"Insufficient data for {ticker}: {len(data)} rows")
        # Convert to Series if it's a DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.squeeze()
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Fetch data for SPY, TLT, GLD
spy_data = fetch_data('SPY', start_date, end_date)
tlt_data = fetch_data('TLT', start_date, end_date)
gld_data = fetch_data('GLD', start_date, end_date)

# Check if data was successfully fetched
if any(data is None for data in [spy_data, tlt_data, gld_data]):
    print("Data fetch failed for yfinance. Exiting.")
    exit(1)

# Fetch 3-month T-bill rates from FRED
try:
    cash_yield = pdr.get_data_fred('TB3MS', start=start_date, end=end_date)
    if cash_yield.empty or len(cash_yield) < 2:
        raise ValueError(f"Insufficient data for TB3MS: {len(cash_yield)} rows")
except Exception as e:
    print(f"Error fetching TB3MS from FRED: {e}")
    exit(1)

# Debug: Print data lengths and first few rows
print(f"SPY data points: {len(spy_data)}")
print("SPY data head:\n", spy_data.head())
print(f"TLT data points: {len(tlt_data)}")
print("TLT data head:\n", tlt_data.head())
print(f"GLD data points: {len(gld_data)}")
print("GLD data head:\n", gld_data.head())
print(f"TB3MS data points: {len(cash_yield)}")
print("TB3MS data head:\n", cash_yield.head())

# Calculate monthly returns manually: (price_t / price_{t-1} - 1)
def calc_manual_returns(series):
    returns = (series / series.shift(1) - 1).dropna()
    return returns

# Compute monthly returns - let pandas align the indices naturally
returns = pd.DataFrame({
    'stocks': calc_manual_returns(spy_data),
    'bonds': calc_manual_returns(tlt_data),
    'gold': calc_manual_returns(gld_data)
}).dropna()

# Convert T-bill annual yield to monthly return (yield / 100 / 12)
cash_returns = (cash_yield['TB3MS'] / 100 / 12).reindex(returns.index).ffill()

# Add cash to returns DataFrame
returns['cash'] = cash_returns

# Filter from 2005 onwards
returns = returns[returns.index >= pd.to_datetime('2005-01-01')]

# Check if returns DataFrame is empty
if returns.empty:
    print("No valid data after processing. Check date range or ticker data.")
    exit(1)

# Debug: Print first few rows of returns
print("\nReturns DataFrame head:")
print(returns.head())
print(f"\nTotal periods: {len(returns)}")

# Initial investment and allocation
initial = 40000.0
alloc = 0.25
assets = ['stocks', 'bonds', 'gold', 'cash']

# Initialize portfolio tracking with one extra row for initial value
port_index = returns.index
port_values = pd.Series(index=port_index, dtype=float, name='Portfolio')
bench_values = pd.DataFrame(index=port_index, columns=['SP500', 'Gold', 'Bonds', 'Cash'])

# Initialize asset values
asset_values = {a: initial * alloc for a in assets}

# Benchmark trackers
bench_trackers = {
    'SP500': initial,
    'Gold': initial,
    'Bonds': initial,
    'Cash': initial
}

# Simulate portfolio with annual rebalancing
for i, date in enumerate(port_index):
    # Update asset values with returns
    for a in assets:
        asset_values[a] *= (1 + returns.loc[date, a])
    
    # Portfolio value after returns
    total = sum(asset_values.values())
    port_values.loc[date] = total
    
    # Update benchmarks
    bench_trackers['SP500'] *= (1 + returns.loc[date, 'stocks'])
    bench_trackers['Gold'] *= (1 + returns.loc[date, 'gold'])
    bench_trackers['Bonds'] *= (1 + returns.loc[date, 'bonds'])
    bench_trackers['Cash'] *= (1 + returns.loc[date, 'cash'])
    
    bench_values.loc[date] = [bench_trackers['SP500'], bench_trackers['Gold'], 
                               bench_trackers['Bonds'], bench_trackers['Cash']]
    
    # Rebalance if end of year (December)
    if date.month == 12:
        for a in assets:
            asset_values[a] = total * alloc

# Compute monthly returns for portfolio
port_monthly_returns = port_values.pct_change().dropna()

# Compute drawdowns for portfolio
cum_max = port_values.cummax()
port_drawdowns = (port_values - cum_max) / cum_max

# Performance metrics
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)

final_port = port_values.iloc[-1]
total_return_port = (final_port / initial - 1) * 100
years = len(returns) / 12
cagr_port = (final_port / initial) ** (1/years) - 1

print(f"\nPortfolio (25/25/25/25):")
print(f"  Final Value: ${final_port:,.2f}")
print(f"  Total Return: {total_return_port:.2f}%")
print(f"  CAGR: {cagr_port*100:.2f}%")
print(f"  Max Drawdown: {port_drawdowns.min()*100:.2f}%")
print(f"  Volatility (annual): {port_monthly_returns.std() * (12**0.5) * 100:.2f}%")

for name, col in [('S&P 500', 'SP500'), ('Gold', 'Gold'), ('Bonds', 'Bonds'), ('Cash', 'Cash')]:
    final = bench_values[col].iloc[-1]
    total_ret = (final / initial - 1) * 100
    cagr = (final / initial) ** (1/years) - 1
    dd = ((bench_values[col] - bench_values[col].cummax()) / bench_values[col].cummax()).min()
    
    print(f"\n{name}:")
    print(f"  Final Value: ${final:,.2f}")
    print(f"  Total Return: {total_ret:.2f}%")
    print(f"  CAGR: {cagr*100:.2f}%")
    print(f"  Max Drawdown: {dd*100:.2f}%")

# Visualizations
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

# 1. Cumulative value over time
plt.figure(figsize=(14, 7))
plt.plot(port_values, label='Portfolio (25/25/25/25)', linewidth=2.5, color='black')
plt.plot(bench_values['SP500'], label='S&P 500', linestyle='--', linewidth=1.5, alpha=0.8)
plt.plot(bench_values['Gold'], label='Gold', linestyle='--', linewidth=1.5, alpha=0.8)
plt.plot(bench_values['Bonds'], label='Bonds (TLT)', linestyle='--', linewidth=1.5, alpha=0.8)
plt.plot(bench_values['Cash'], label='Cash (T-Bills)', linestyle='--', linewidth=1.5, alpha=0.8)
plt.title('Harry Browne Permanent Portfolio: Cumulative Growth', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Portfolio Value ($)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Portfolio drawdowns
plt.figure(figsize=(14, 6))
plt.plot(port_drawdowns * 100, color='red', linewidth=2)
plt.title('Portfolio Drawdowns Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Drawdown (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.fill_between(port_drawdowns.index, port_drawdowns * 100, 0, color='red', alpha=0.2)
plt.tight_layout()
plt.show()

# 3. Rolling 12-month returns
plt.figure(figsize=(14, 6))
rolling_returns = port_values.pct_change(12) * 100
rolling_returns.plot(color='blue', linewidth=1.5)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
plt.title('Portfolio Rolling 12-Month Returns', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('12-Month Return (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
