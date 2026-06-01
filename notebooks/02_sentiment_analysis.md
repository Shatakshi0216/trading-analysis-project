````markdown
# 02_Sentiment_Analysis

Comprehensive analysis of Bitcoin market sentiment and its temporal patterns.

## Sentiment Trend Analysis

```python
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.analysis import TradingAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

loader = DataLoader()
sentiment_df, _ = loader.load_all_data()
sentiment_df = DataPreprocessor.preprocess_sentiment_data(sentiment_df)

# Analyze trends
trends = TradingAnalyzer.analyze_sentiment_trends(sentiment_df)
print("Volatility Metrics:")
for key, value in trends['volatility'].items():
    print(f"  {key}: {value:.2f}")
```

## Sentiment Regimes

### Classification Distribution

```python
regime_counts = sentiment_df['sentiment_category'].value_counts()
print("Sentiment Regime Distribution:")
print(regime_counts)
print("\nPercentage Distribution:")
print((regime_counts / len(sentiment_df) * 100).round(2))
```

### Regime Transitions

```python
# Identify transitions
sentiment_df['prev_category'] = sentiment_df['sentiment_category'].shift(1)
transitions = sentiment_df[sentiment_df['sentiment_category'] != sentiment_df['prev_category']]
print(f"Number of Sentiment Transitions: {len(transitions)}")
print(f"Average Days Between Transitions: {len(sentiment_df) / len(transitions):.1f}")
```

## Statistical Properties

### Volatility Analysis

```python
# Monthly volatility
sentiment_df['year_month'] = sentiment_df['date'].dt.to_period('M')
monthly_vol = sentiment_df.groupby('year_month')['fear_greed_value'].std()
print("Monthly Volatility:")
print(monthly_vol)

# Plot volatility
fig, ax = plt.subplots(figsize=(12, 5))
monthly_vol.plot(ax=ax, marker='o')
ax.set_xlabel('Month')
ax.set_ylabel('Volatility (Std Dev)')
ax.set_title('Sentiment Volatility by Month')
plt.tight_layout()
plt.show()
```

## Extreme Events

### Identifying Extreme Sentiment

```python
extreme_fear = sentiment_df[sentiment_df['fear_greed_value'] < 25]
extreme_greed = sentiment_df[sentiment_df['fear_greed_value'] > 75]

print("Extreme Fear Events:")
print(extreme_fear[['date', 'fear_greed_value', 'classification']])
print(f"\nTotal Extreme Fear Days: {len(extreme_fear)}")
print(f"Total Extreme Greed Days: {len(extreme_greed)}")
```

## Key Findings

1. **Sentiment Mean**: {mean_sentiment}
2. **Sentiment Std Dev**: {std_sentiment}
3. **Most Common Regime**: {mode_regime}
4. **Transition Frequency**: {transition_freq}

## Insights for Traders

- **Extreme Sentiment Periods**: Often precede significant market moves
- **Volatility Clustering**: High volatility tends to persist
- **Regime Persistence**: Sentiment tends to stay in similar regimes

## Next Steps

- Merge sentiment with trader performance data
- Analyze trader behavior during different sentiment regimes
- Build predictive models for trader success
````
