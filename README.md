# Trading Analysis Project: Bitcoin Sentiment & Hyperliquid Trader Performance

A comprehensive data science project analyzing the relationship between Bitcoin market sentiment and trader performance on Hyperliquid. This project uncovers hidden patterns in trading behavior and delivers actionable insights for smarter trading strategies.

## 📊 Project Overview

This project explores two primary datasets:
1. **Bitcoin Market Sentiment Dataset** - Fear/Greed sentiment indicators over time
2. **Hyperliquid Historical Trader Data** - Detailed trader performance metrics including execution prices, positions, leverage, and trading events

### Objectives
- Analyze the correlation between market sentiment and trader performance
- Identify profitable trading patterns and risk factors
- Uncover hidden patterns in trader behavior across different market conditions
- Deliver actionable insights for trading strategy optimization

## 📈 Datasets

### 1. Bitcoin Market Sentiment Data
- **Source**: Fear/Greed Index
- **Features**: 
  - Date: Temporal dimension
  - Classification: Fear, Greed, or Neutral sentiment
  - Sentiment Score: Quantitative measure

### 2. Hyperliquid Historical Trader Data
- **Source**: Hyperliquid Trading Platform
- **Features**:
  - Account information and trading symbols
  - Execution prices and order sizes
  - Position details (side, leverage, margin)
  - Trading events and performance metrics
  - Time-series trading activity data

## 🔧 Tech Stack

- **Python 3.8+**
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning models
- **Matplotlib & Seaborn** - Data visualization
- **Plotly** - Interactive visualizations
- **Jupyter Notebook** - Exploratory analysis

## 📁 Project Structure

```
trading-analysis-project/
├── README.md
├── requirements.txt
├── data/
│   ├── bitcoin_sentiment.csv
│   └── hyperliquid_traders.csv
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_trader_performance_analysis.ipynb
│   ├── 04_correlation_and_patterns.ipynb
│   └── 05_predictive_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_preprocessor.py
│   ├── analysis.py
│   ├── visualization.py
│   └── models.py
├── outputs/
│   ├── visualizations/
│   └── reports/
└── tests/
    └── test_analysis.py
```

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/Shatakshi0216/trading-analysis-project.git
cd trading-analysis-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Launch Jupyter and open notebooks
jupyter notebook

# Or run the analysis pipeline
python src/main.py
```

## 📊 Key Analyses

### 1. Exploratory Data Analysis (EDA)
- Data quality assessment and missing value analysis
- Sentiment distribution and temporal patterns
- Trader performance metrics overview
- Statistical summaries and correlations

### 2. Sentiment Analysis
- Fear/Greed index trends over time
- Sentiment regime identification
- Impact of sentiment on market volatility
- Sentiment-based trading opportunities

### 3. Trader Performance Analysis
- Win rate and profitability metrics
- Risk-adjusted returns (Sharpe ratio, Sortino ratio)
- Position sizing and leverage usage patterns
- Trading frequency and time-of-day effects

### 4. Correlation & Pattern Analysis
- Sentiment-Performance correlation
- Temporal patterns in trading behavior
- Market condition clustering
- Sentiment-driven trader behavior changes

### 5. Predictive Modeling
- Machine Learning models for trader success prediction
- Feature importance analysis
- Model performance evaluation
- Cross-validation and robustness testing

## 💡 Key Findings (To be populated)

*Results and insights from analysis will be documented here*

## 📈 Visualizations

The project generates:
- Time-series sentiment trends with confidence intervals
- Trader performance distribution and box plots
- Heatmaps of correlations
- Interactive Plotly dashboards
- ROC curves and confusion matrices for ML models

## 🎯 Business Impact

- **Strategy Optimization**: Identify when sentiment aligns with profitable trades
- **Risk Management**: Understand which sentiment conditions lead to losses
- **Trader Success**: Predict trader profitability based on market conditions
- **Market Insights**: Discover latent patterns in collective trading behavior

## 🔍 Methodology

1. **Data Collection & Loading**: Integrate both datasets
2. **Data Cleaning**: Handle missing values, outliers, and data quality issues
3. **Feature Engineering**: Create meaningful features for analysis
4. **Exploratory Analysis**: Understand data distributions and relationships
5. **Statistical Analysis**: Test hypotheses and measure correlations
6. **Machine Learning**: Build predictive models
7. **Visualization & Reporting**: Create compelling visualizations and insights

## 📝 Files Description

- `data_loader.py` - Functions to load and validate data
- `data_preprocessor.py` - Data cleaning and feature engineering
- `analysis.py` - Statistical analysis and correlation studies
- `visualization.py` - All plotting and visualization functions
- `models.py` - Machine learning model implementations
- `main.py` - Main execution pipeline

## 🧪 Testing

Run tests to validate the analysis pipeline:
```bash
pytest tests/
```

## 📚 References

- Bitcoin Fear & Greed Index: https://alternative.me/crypto/fear-and-greed-index/
- Hyperliquid Platform: https://hyperliquid.xyz/
- Trading Strategy Research: Academic papers on market sentiment and trader behavior

## 👤 Author

Shatakshi0216

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

---

**Note**: This project demonstrates data analysis, machine learning, and business intelligence skills essential for data science roles in fintech and trading companies.
