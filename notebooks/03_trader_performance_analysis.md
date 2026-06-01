````markdown
# 03_Trader_Performance_Analysis

Deep dive into Hyperliquid trader performance metrics and risk analysis.

## Performance Overview

```python
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.analysis import TradingAnalyzer
import pandas as pd
import numpy as np

loader = DataLoader()
_, trader_df = loader.load_all_data()
trader_df = DataPreprocessor.preprocess_trader_data(trader_df)

# Get performance metrics
perf = TradingAnalyzer.analyze_trader_performance(trader_df)

print("=" * 50)
print("TRADER PERFORMANCE METRICS")
print("=" * 50)
for key, value in perf.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")
```

## Risk Metrics

### Drawdown Analysis

```python
# Calculate running maximum and drawdown
trader_df['cumulative_pnl'] = trader_df['pnl'].cumsum()
trader_df['running_max'] = trader_df['cumulative_pnl'].expanding().max()
trader_df['drawdown'] = trader_df['cumulative_pnl'] - trader_df['running_max']
trader_df['drawdown_pct'] = (trader_df['drawdown'] / trader_df['running_max'].abs() * 100).fillna(0)

print("Maximum Drawdown:", trader_df['drawdown'].min())
print("Maximum Drawdown %:", trader_df['drawdown_pct'].min())
print("Average Drawdown:", trader_df['drawdown'].mean())
```

## Symbol-wise Analysis

### Performance by Trading Pair

```python
symbol_stats = trader_df.groupby('symbol').agg({
    'pnl': ['sum', 'mean', 'std', 'count'],
    'is_profitable': lambda x: (x == 1).sum() / len(x),
    'leverage': 'mean'
}).round(2)

print("Performance by Symbol:")
print(symbol_stats)

# Plot symbol performance
fig, ax = plt.subplots(figsize=(12, 6))
symbol_pnl = trader_df.groupby('symbol')['pnl'].sum().sort_values()
symbol_pnl.plot(kind='barh', ax=ax)
ax.set_xlabel('Total PnL')
ax.set_title('Cumulative PnL by Trading Pair')
plt.tight_layout()
plt.show()
```

## Side Analysis (Long vs Short)

### Directional Performance

```python
side_analysis = trader_df.groupby('side').agg({
    'pnl': ['count', 'sum', 'mean', 'std'],
    'is_profitable': 'mean',
    'leverage': 'mean'
}).round(3)

print("Performance by Side (Long/Short):")
print(side_analysis)
```

## Account Analysis

### Multi-account Performance

```python
account_stats = trader_df.groupby('account').agg({
    'pnl': ['sum', 'mean', 'count'],
    'is_profitable': 'mean',
    'leverage': 'mean'
}).round(2)

print("Performance by Account:")
print(account_stats)
```

## Time Pattern Analysis

### Intraday Performance

```python
hourly_perf = trader_df.groupby('hour').agg({
    'pnl': ['mean', 'std', 'count'],
    'is_profitable': 'mean'
}).round(2)

print("Performance by Hour of Day:")
print(hourly_perf)

# Visualize
fig, ax = plt.subplots(figsize=(14, 5))
hourly_avg = trader_df.groupby('hour')['pnl'].mean()
hourly_avg.plot(kind='bar', ax=ax, color='steelblue')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average PnL')
ax.set_title('Average PnL by Hour')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

## Key Performance Indicators

### Summary Statistics

```python
print("\nKey Performance Summary:")
print(f"Total Trades: {len(trader_df)}")
print(f"Winning Trades: {(trader_df['pnl'] > 0).sum()}")
print(f"Losing Trades: {(trader_df['pnl'] < 0).sum()}")
print(f"Win Rate: {perf['win_rate']:.2%}")
print(f"Profit Factor: {abs(trader_df[trader_df['pnl'] > 0]['pnl'].sum() / trader_df[trader_df['pnl'] < 0]['pnl'].sum()):.2f}")
print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
```

## Insights

- **Best Performing Symbol**: {best_symbol}
- **Worst Performing Symbol**: {worst_symbol}
- **Best Trading Hour**: {best_hour}
- **Win Rate Trend**: {win_rate_trend}

## Next Steps

1. Merge with sentiment data
2. Analyze sentiment-performance correlation
3. Identify optimal trading conditions
4. Build predictive models
````
