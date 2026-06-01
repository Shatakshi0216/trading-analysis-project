"""Main execution pipeline for trading analysis."""
import sys
sys.path.insert(0, '.')

from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.analysis import TradingAnalyzer
from src.models import TradingModels
import pandas as pd

def main():
    """Execute main analysis pipeline."""
    print("=" * 60)
    print("Trading Analysis Pipeline")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    loader = DataLoader()
    sentiment_df, trader_df = loader.load_all_data()
    print(loader.get_data_summary())
    
    # Preprocess data
    print("\n[2] Preprocessing data...")
    sentiment_df = DataPreprocessor.preprocess_sentiment_data(sentiment_df)
    trader_df = DataPreprocessor.preprocess_trader_data(trader_df)
    
    # Merge datasets
    print("\n[3] Merging datasets...")
    merged_df = DataPreprocessor.merge_datasets(sentiment_df, trader_df)
    merged_df = DataPreprocessor.engineer_features(merged_df)
    
    # Analyze
    print("\n[4] Running analysis...")
    sentiment_trends = TradingAnalyzer.analyze_sentiment_trends(sentiment_df)
    trader_perf = TradingAnalyzer.analyze_trader_performance(trader_df)
    correlation = TradingAnalyzer.analyze_sentiment_trader_correlation(merged_df)
    patterns = TradingAnalyzer.identify_patterns(merged_df)
    tests = TradingAnalyzer.statistical_tests(merged_df)
    
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    print("\nTrader Performance:")
    for key, value in trader_perf.items():
        print(f"  {key}: {value}")
    
    print("\nCorrelation Analysis:")
    for key, value in correlation.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("✓ Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()