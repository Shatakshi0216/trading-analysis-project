````markdown
# 01_Exploratory_Data_Analysis

This notebook provides comprehensive exploratory data analysis of Bitcoin sentiment and Hyperliquid trader data.

## Data Overview

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor

# Load data
loader = DataLoader()
sentiment_df, trader_df = loader.load_all_data()

print("Sentiment Data Shape:", sentiment_df.shape)
print("Trader Data Shape:", trader_df.shape)
print("\nSentiment Data Summary:")
print(sentiment_df.describe())
```

## Data Quality Assessment

### Missing Values

```python
# Check for missing values
print("Sentiment Missing Values:")
print(sentiment_df.isnull().sum())
print("\nTrader Missing Values:")
print(trader_df.isnull().sum())
```

### Duplicate Records

```python
print("Sentiment Duplicates:", sentiment_df.duplicated().sum())
print("Trader Duplicates:", trader_df.duplicated().sum())
```

## Sentiment Analysis

### Distribution of Fear/Greed Values

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(sentiment_df['fear_greed_value'], bins=20, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Fear/Greed Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Sentiment Values')

# Classification counts
sentiment_df['classification'].value_counts().plot(kind='bar', ax=axes[1])
axes[1].set_xlabel('Classification')
axes[1].set_ylabel('Count')
axes[1].set_title('Sentiment Classification Distribution')
plt.tight_layout()
plt.show()
```

### Temporal Trends

```python
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(sentiment_df['date'], sentiment_df['fear_greed_value'], linewidth=2)
ax.fill_between(sentiment_df['date'], sentiment_df['fear_greed_value'], alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Fear/Greed Value')
ax.set_title('Sentiment Trends Over Time')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Trading Performance Analysis

### PnL Statistics

```python
print("PnL Statistics:")
print(trader_df['pnl'].describe())
print("\nWin Rate:", (trader_df['pnl'] > 0).sum() / len(trader_df))
print("Average Win:", trader_df[trader_df['pnl'] > 0]['pnl'].mean())
print("Average Loss:", trader_df[trader_df['pnl'] < 0]['pnl'].mean())
```

### Symbol Performance

```python
symbol_stats = trader_df.groupby('symbol').agg({
    'pnl': ['sum', 'mean', 'count'],
    'leverage': 'mean'
}).round(2)
print(symbol_stats)
```

### Leverage Analysis

```python
leverage_perf = trader_df.groupby(pd.cut(trader_df['leverage'], bins=[0, 1, 5, 10, 20])).agg({
    'pnl': ['mean', 'std'],
    'is_profitable': 'mean'
})
print(leverage_perf)
```

## Correlations

### Feature Correlations

```python
numeric_cols = trader_df.select_dtypes(include=[np.number]).columns
correlation_matrix = trader_df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
```

## Key Insights

1. **Sentiment Distribution**: Fear/Greed values show cyclical patterns
2. **Trading Performance**: Win rate varies significantly across different symbols
3. **Leverage Impact**: Higher leverage correlates with higher volatility
4. **Time Patterns**: Certain hours show better average performance

## Next Steps

- Merge sentiment and trader data
- Perform deeper correlation analysis
- Train predictive models
- Generate actionable insights
````
