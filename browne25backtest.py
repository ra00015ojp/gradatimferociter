import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
import datetime

st.set_page_config(page_title="Permanent Portfolio Analyzer", layout="wide")

st.title("📊 Harry Browne Permanent Portfolio Analyzer")
st.markdown("Analyze the performance of a balanced 25/25/25/25 portfolio across stocks, bonds, gold, and cash")

# Sidebar controls
st.sidebar.header("Portfolio Settings")

# Years of analysis slider
years_back = st.sidebar.slider(
    "Years of Historical Data",
    min_value=5,
    max_value=20,
    value=15,
    step=1,
    help="Select how many years of historical data to analyze"
)

# Initial investment input
initial_investment = st.sidebar.number_input(
    "Initial Investment ($)",
    min_value=1000,
    max_value=10000000,
    value=40000,
    step=1000,
    help="Enter your initial investment amount"
)

# Asset allocation inputs
st.sidebar.subheader("Asset Allocation (%)")
col1, col2 = st.sidebar.columns(2)
with col1:
    alloc_stocks = st.number_input("Stocks", min_value=0, max_value=100, value=25, step=5)
    alloc_bonds = st.number_input("Bonds", min_value=0, max_value=100, value=25, step=5)
with col2:
    alloc_gold = st.number_input("Gold", min_value=0, max_value=100, value=25, step=5)
    alloc_cash = st.number_input("Cash", min_value=0, max_value=100, value=25, step=5)

# Check if allocation sums to 100
total_alloc = alloc_stocks + alloc_bonds + alloc_gold + alloc_cash
if total_alloc != 100:
    st.sidebar.error(f"⚠️ Allocation must sum to 100% (currently {total_alloc}%)")
    st.stop()

# Calculate start date
end_date = datetime.date.today()
start_date = (end_date - datetime.timedelta(days=years_back*365 + 180)).strftime('%Y-%m-%d')

# Function to fetch data with caching
@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, interval='1mo', auto_adjust=False, progress=False)['Adj Close']
        if data.empty or len(data) < 2:
            raise ValueError(f"Insufficient data for {ticker}")
        if isinstance(data, pd.DataFrame):
            data = data.squeeze()
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_tbill_data(start, end):
    try:
        cash_yield = pdr.get_data_fred('TB3MS', start=start, end=end)
        if cash_yield.empty or len(cash_yield) < 2:
            raise ValueError("Insufficient data for TB3MS")
        return cash_yield
    except Exception as e:
        st.error(f"Error fetching T-Bill data: {e}")
        return None

# Fetch data
with st.spinner("Fetching market data..."):
    spy_data = fetch_data('SPY', start_date, end_date.strftime('%Y-%m-%d'))
    tlt_data = fetch_data('TLT', start_date, end_date.strftime('%Y-%m-%d'))
    gld_data = fetch_data('GLD', start_date, end_date.strftime('%Y-%m-%d'))
    cash_yield = fetch_tbill_data(start_date, end_date.strftime('%Y-%m-%d'))

# Check if data was successfully fetched
if any(data is None for data in [spy_data, tlt_data, gld_data, cash_yield]):
    st.error("Failed to fetch required market data. Please try again later.")
    st.stop()

# Calculate monthly returns
def calc_manual_returns(series):
    returns = (series / series.shift(1) - 1).dropna()
    return returns

returns = pd.DataFrame({
    'stocks': calc_manual_returns(spy_data),
    'bonds': calc_manual_returns(tlt_data),
    'gold': calc_manual_returns(gld_data)
}).dropna()

# Convert T-bill annual yield to monthly return
cash_returns = (cash_yield['TB3MS'] / 100 / 12).reindex(returns.index).ffill()
returns['cash'] = cash_returns

# Filter from start of analysis period
min_date = pd.to_datetime(start_date)
returns = returns[returns.index >= min_date]

if returns.empty:
    st.error("No valid data after processing. Try adjusting the date range.")
    st.stop()

# Portfolio simulation
initial = initial_investment
alloc_dict = {
    'stocks': alloc_stocks / 100,
    'bonds': alloc_bonds / 100,
    'gold': alloc_gold / 100,
    'cash': alloc_cash / 100
}
assets = ['stocks', 'bonds', 'gold', 'cash']

port_index = returns.index
port_values = pd.Series(index=port_index, dtype=float, name='Portfolio')
bench_values = pd.DataFrame(index=port_index, columns=['SP500', 'Gold', 'Bonds', 'Cash'])

# Initialize asset values
asset_values = {a: initial * alloc_dict[a] for a in assets}

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
            asset_values[a] = total * alloc_dict[a]

# Compute metrics
port_monthly_returns = port_values.pct_change().dropna()
cum_max = port_values.cummax()
port_drawdowns = (port_values - cum_max) / cum_max

# Display metrics
st.header("📈 Performance Summary")

final_port = port_values.iloc[-1]
total_return_port = (final_port / initial - 1) * 100
years = len(returns) / 12
cagr_port = (final_port / initial) ** (1/years) - 1

# Main portfolio metrics in columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Final Value", f"${final_port:,.2f}", f"{total_return_port:.1f}%")
with col2:
    st.metric("CAGR", f"{cagr_port*100:.2f}%")
with col3:
    st.metric("Max Drawdown", f"{port_drawdowns.min()*100:.2f}%")
with col4:
    st.metric("Annual Volatility", f"{port_monthly_returns.std() * (12**0.5) * 100:.2f}%")

# Benchmark comparison
st.subheader("Benchmark Comparison")
bench_metrics = []
for name, col in [('S&P 500', 'SP500'), ('Gold', 'Gold'), ('Bonds', 'Bonds'), ('Cash', 'Cash')]:
    final = bench_values[col].iloc[-1]
    total_ret = (final / initial - 1) * 100
    cagr = (final / initial) ** (1/years) - 1
    dd = ((bench_values[col] - bench_values[col].cummax()) / bench_values[col].cummax()).min()
    bench_metrics.append({
        'Asset': name,
        'Final Value': f"${final:,.2f}",
        'Total Return': f"{total_ret:.2f}%",
        'CAGR': f"{cagr*100:.2f}%",
        'Max Drawdown': f"{dd*100:.2f}%"
    })

st.dataframe(pd.DataFrame(bench_metrics), hide_index=True, use_container_width=True)

# Visualizations
st.header("📊 Visualizations")

# 1. Cumulative value over time
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(port_values, label=f'Portfolio ({alloc_stocks}/{alloc_bonds}/{alloc_gold}/{alloc_cash})', linewidth=2.5, color='black')
ax1.plot(bench_values['SP500'], label='S&P 500', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.plot(bench_values['Gold'], label='Gold', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.plot(bench_values['Bonds'], label='Bonds (TLT)', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.plot(bench_values['Cash'], label='Cash (T-Bills)', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.set_title('Cumulative Growth Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig1)

# 2. Portfolio drawdowns
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(port_drawdowns * 100, color='red', linewidth=2)
ax2.set_title('Portfolio Drawdowns Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Drawdown (%)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.fill_between(port_drawdowns.index, port_drawdowns * 100, 0, color='red', alpha=0.2)
plt.tight_layout()
st.pyplot(fig2)

# 3. Rolling 12-month returns
fig3, ax3 = plt.subplots(figsize=(12, 5))
rolling_returns = port_values.pct_change(12) * 100
ax3.plot(rolling_returns, color='blue', linewidth=1.5)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax3.set_title('Portfolio Rolling 12-Month Returns', fontsize=14, fontweight='bold')
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('12-Month Return (%)', fontsize=12)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)

# Footer
st.markdown("---")
st.markdown("*Data sources: Yahoo Finance (SPY, TLT, GLD) and FRED (3-Month T-Bills). Portfolio rebalances annually in December.*")
