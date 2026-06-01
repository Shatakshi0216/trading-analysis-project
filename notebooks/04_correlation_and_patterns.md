````markdown
# 04_Correlation_and_Patterns

Analysis of correlations between sentiment and trader performance, plus pattern discovery.

## Sentiment-Performance Correlation

```python
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.analysis import TradingAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
loader = DataLoader()
sentiment_df, trader_df = loader.load_all_data()
sentiment_df = DataPreprocessor.preprocess_sentiment_data(sentiment_df)
trader_df = DataPreprocessor.preprocess_trader_data(trader_df)

# Merge datasets
merged_df = DataPreprocessor.merge_datasets(sentiment_df, trader_df)
merged_df = DataPreprocessor.engineer_features(merged_df)

# Analyze correlation
correlation_analysis = TradingAnalyzer.analyze_sentiment_trader_correlation(merged_df)

print("=" * 60)
print("SENTIMENT-PERFORMANCE CORRELATION ANALYSIS")
print("=" * 60)
print(f"\nOverall Correlation: {correlation_analysis['overall_correlation']:.3f}")
```

## Sentiment Regime Performance

### Fear Regime Analysis

```python
fear_data = correlation_analysis['fear_regime']
print("\nFEAR REGIME (Fear/Greed < 45):")
print(f"  Trades: {fear_data['trades']}")
print(f"  Avg PnL: ${fear_data['avg_pnl']:.2f}")
print(f"  Win Rate: {fear_data['win_rate']:.2%}")
print(f"  Volatility: {fear_data['volatility']:.2f}")
```

### Neutral Regime Analysis

```python
neutral_data = correlation_analysis['neutral_regime']
print("\nNEUTRAL REGIME (45 <= Fear/Greed <= 55):")
print(f"  Trades: {neutral_data['trades']}")
print(f"  Avg PnL: ${neutral_data['avg_pnl']:.2f}")
print(f"  Win Rate: {neutral_data['win_rate']:.2%}")
print(f"  Volatility: {neutral_data['volatility']:.2f}")
```

### Greed Regime Analysis

```python
greed_data = correlation_analysis['greed_regime']
print("\nGREED REGIME (Fear/Greed > 55):")
print(f"  Trades: {greed_data['trades']}")
print(f"  Avg PnL: ${greed_data['avg_pnl']:.2f}")
print(f"  Win Rate: {greed_data['win_rate']:.2%}")
print(f"  Volatility: {greed_data['volatility']:.2f}")
```

## Visualization: Sentiment vs Performance

```python
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(merged_df['fear_greed_value'], merged_df['pnl'], 
                     c=merged_df['pnl'], cmap='RdYlGn', alpha=0.6, s=50)

# Add trend line
valid_idx = ~(merged_df['fear_greed_value'].isna() | merged_df['pnl'].isna())
z = np.polyfit(merged_df.loc[valid_idx, 'fear_greed_value'], 
               merged_df.loc[valid_idx, 'pnl'], 1)
p = np.poly1d(z)
x_trend = np.linspace(merged_df['fear_greed_value'].min(), 
                      merged_df['fear_greed_value'].max(), 100)
ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend')

ax.set_xlabel('Fear/Greed Index', fontsize=12)
ax.set_ylabel('PnL ($)', fontsize=12)
ax.set_title('Sentiment vs Trading Performance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
cbar = plt.colorbar(scatter)
cbar.set_label('PnL ($)', fontsize=12)
plt.tight_layout()
plt.show()
```

## Pattern Discovery

### Hourly Patterns

```python
patterns = TradingAnalyzer.identify_patterns(merged_df)

hourly_perf = merged_df.groupby('hour').agg({
    'pnl': ['mean', 'count'],
    'is_profitable': 'mean'
}).round(2)

print("\nHOURLY TRADING PATTERNS:")
print(hourly_perf)

# Visualize
fig, ax = plt.subplots(figsize=(14, 5))
merged_df.groupby('hour')['pnl'].mean().plot(kind='bar', ax=ax, color='steelblue')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average PnL ($)')
ax.set_title('Average PnL by Hour of Day', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

### Leverage Patterns

```python
leverage_impact = patterns['leverage_impact']
print("\nLEVERAGE IMPACT ANALYSIS:")
print(f"High Leverage (>10x) Avg PnL: ${leverage_impact['high_leverage_avg_pnl']:.2f}")
print(f"High Leverage Win Rate: {leverage_impact['high_leverage_win_rate']:.2%}")
print(f"Low Leverage (<=10x) Avg PnL: ${leverage_impact['low_leverage_avg_pnl']:.2f}")
print(f"Low Leverage Win Rate: {leverage_impact['low_leverage_win_rate']:.2%}")
```

## Feature Correlations

### Correlation Matrix

```python
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
corr_matrix = merged_df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Statistical Significance Tests

```python
tests = TradingAnalyzer.statistical_tests(merged_df)

if 'correlation_significance' in tests:
    corr_test = tests['correlation_significance']
    print("\nCORRELATION SIGNIFICANCE TEST:")
    print(f"Correlation Coefficient: {corr_test['correlation']:.3f}")
    print(f"P-value: {corr_test['p_value']:.6f}")
    print(f"Statistically Significant: {'Yes' if corr_test['significant'] else 'No'}")
```

## Key Insights

1. **Sentiment Impact on Performance**: Different sentiment regimes show distinct win rates
2. **Optimal Trading Hours**: Certain hours consistently outperform others
3. **Leverage Risk**: Higher leverage increases both gains and losses
4. **Regime-Specific Strategies**: Tailor approach based on market sentiment

## Actionable Recommendations

- During Fear periods: Consider defensive strategies
- During Greed periods: Opportunity for aggressive positions
- Best trading hours: Focus capital during peak performance windows
- Risk Management: Adjust leverage based on sentiment regime
````
