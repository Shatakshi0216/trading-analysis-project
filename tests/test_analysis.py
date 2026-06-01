"""Unit tests for trading analysis modules."""
import unittest
import pandas as pd
import numpy as np
from src.analysis import TradingAnalyzer
from src.data_preprocessor import DataPreprocessor

class TestTradingAnalyzer(unittest.TestCase):
    """Test cases for TradingAnalyzer."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_df = pd.DataFrame({
            'fear_greed_value': [30, 45, 60, 75, 50],
            'pnl': [100, -50, 200, 150, 25],
            'is_profitable': [1, 0, 1, 1, 1],
            'leverage': [5, 3, 10, 2, 8],
            'hour': [8, 9, 10, 11, 12],
            'day_of_week': [0, 0, 0, 0, 1],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='H')
        })
    
    def test_trader_performance_analysis(self):
        """Test trader performance analysis."""
        result = TradingAnalyzer.analyze_trader_performance(self.sample_df)
        
        self.assertIn('total_trades', result)
        self.assertIn('win_rate', result)
        self.assertIn('total_pnl', result)
        self.assertEqual(result['total_trades'], 5)
        self.assertGreater(result['win_rate'], 0)
    
    def test_correlation_analysis(self):
        """Test correlation analysis."""
        result = TradingAnalyzer.analyze_sentiment_trader_correlation(self.sample_df)
        
        self.assertIn('overall_correlation', result)
        self.assertIn('fear_regime', result)
        self.assertIn('greed_regime', result)
    
    def test_sentiment_trends(self):
        """Test sentiment trend analysis."""
        sample_sentiment = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'fear_greed_value': [30, 45, 60, 75, 50],
            'sentiment_category': ['Fear', 'Neutral', 'Greed', 'Extreme Greed', 'Neutral']
        })
        
        result = TradingAnalyzer.analyze_sentiment_trends(sample_sentiment)
        
        self.assertIn('volatility', result)
        self.assertGreater(result['volatility']['range'], 0)

class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor."""
    
    def test_feature_engineering(self):
        """Test feature engineering."""
        df = pd.DataFrame({
            'pnl': [100, -50, 200],
            'leverage': [5, 10, 2],
            'timestamp': pd.date_range('2024-01-01', periods=3),
            'notional_value': [1000, 2000, 1500]
        })
        
        df['is_profitable'] = (df['pnl'] > 0).astype(int)
        self.assertEqual(df['is_profitable'].sum(), 2)

if __name__ == '__main__':
    unittest.main()