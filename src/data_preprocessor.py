"""
Data preprocessor module for trading analysis project.
Cleans, validates, and engineers features for analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict


class DataPreprocessor:
    """Handle data cleaning and feature engineering."""
    
    @staticmethod
    def preprocess_sentiment_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Bitcoin sentiment data.
        
        Args:
            df: Raw sentiment DataFrame
            
        Returns:
            Preprocessed sentiment DataFrame
        """
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
        
        # Fill missing values
        df['fear_greed_value'] = df['fear_greed_value'].fillna(df['fear_greed_value'].mean())
        
        # Clip to valid range [0, 100]
        df['fear_greed_value'] = df['fear_greed_value'].clip(0, 100)
        
        # Create sentiment categories
        def categorize_sentiment(value):
            if value < 25:
                return "Extreme Fear"
            elif value < 45:
                return "Fear"
            elif value < 55:
                return "Neutral"
            elif value < 75:
                return "Greed"
            else:
                return "Extreme Greed"
        
        df['sentiment_category'] = df['fear_greed_value'].apply(categorize_sentiment)
        
        # Extract temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['week'] = df['date'].dt.isocalendar().week
        df['day_of_week'] = df['date'].dt.dayofweek
        
        print(f"✓ Preprocessed sentiment data: {len(df)} records")
        
        return df
    
    @staticmethod
    def preprocess_trader_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Hyperliquid trader data.
        
        Args:
            df: Raw trader DataFrame
            
        Returns:
            Preprocessed trader DataFrame
        """
        df = df.copy()
        
        # Remove rows with missing critical values
        df = df.dropna(subset=['timestamp', 'symbol', 'pnl', 'price', 'size'])
        
        # Ensure numeric columns
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['size'] = pd.to_numeric(df['size'], errors='coerce')
        df['leverage'] = pd.to_numeric(df['leverage'], errors='coerce')
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
        
        # Fill missing leverage with median
        df['leverage'] = df['leverage'].fillna(df['leverage'].median())
        
        # Remove outliers in PnL (> 10 std deviations)
        pnl_mean = df['pnl'].mean()
        pnl_std = df['pnl'].std()
        df = df[df['pnl'] <= pnl_mean + 10 * pnl_std]
        
        # Create features
        df['is_profitable'] = (df['pnl'] > 0).astype(int)
        df['is_long'] = (df['side'].str.lower() == 'long').astype(int)
        df['is_short'] = (df['side'].str.lower() == 'short').astype(int)
        df['notional_value'] = df['price'] * df['size']
        df['pnl_percentage'] = (df['pnl'] / df['notional_value'] * 100).fillna(0)
        
        # Extract temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['date'] = df['timestamp'].dt.date
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        
        print(f"✓ Preprocessed trader data: {len(df)} records")
        
        return df
    
    @staticmethod
    def merge_datasets(sentiment_df: pd.DataFrame, trader_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge sentiment and trader datasets.
        
        Args:
            sentiment_df: Preprocessed sentiment data
            trader_df: Preprocessed trader data
            
        Returns:
            Merged DataFrame
        """
        # Convert timestamps to dates for merging
        trader_df['sentiment_date'] = trader_df['timestamp'].dt.date
        sentiment_df['sentiment_date'] = sentiment_df['date'].dt.date
        
        # Merge on date
        merged = trader_df.merge(
            sentiment_df[['sentiment_date', 'fear_greed_value', 'sentiment_category', 'sentiment_value']],
            left_on='sentiment_date',
            right_on='sentiment_date',
            how='left'
        )
        
        # Forward fill missing sentiment values (in case of missing dates)
        merged = merged.sort_values('timestamp')
        merged['fear_greed_value'] = merged['fear_greed_value'].fillna(method='ffill')
        merged['sentiment_category'] = merged['sentiment_category'].fillna(method='ffill')
        
        print(f"✓ Merged datasets: {len(merged)} records")
        
        return merged
    
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features for analysis.
        
        Args:
            df: Merged DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Rolling statistics
        df['pnl_rolling_mean_7'] = df.groupby('symbol')['pnl'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df['pnl_rolling_std_7'] = df.groupby('symbol')['pnl'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        )
        
        # Cumulative PnL
        df['cumulative_pnl'] = df.groupby('symbol')['pnl'].cumsum()
        
        # Leverage categorization
        def categorize_leverage(lev):
            if lev <= 1:
                return "No Leverage"
            elif lev <= 5:
                return "Low"
            elif lev <= 10:
                return "Medium"
            else:
                return "High"
        
        df['leverage_category'] = df['leverage'].apply(categorize_leverage)
        
        # Trade size categorization
        df['trade_size_category'] = pd.qcut(df['notional_value'], q=3, labels=['Small', 'Medium', 'Large'])
        
        # Market condition (based on sentiment and volatility)
        df['market_condition'] = df['sentiment_category'].apply(
            lambda x: 'Bullish' if x in ['Greed', 'Extreme Greed'] else 'Bearish' if x in ['Fear', 'Extreme Fear'] else 'Neutral'
        )
        
        print(f"✓ Engineered features for {len(df)} records")
        
        return df
    
    @staticmethod
    def get_preprocessing_report(original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict:
        """
        Generate preprocessing report.
        
        Args:
            original_df: Original DataFrame
            processed_df: Processed DataFrame
            
        Returns:
            Dictionary with preprocessing statistics
        """
        report = {
            'original_records': len(original_df),
            'processed_records': len(processed_df),
            'records_removed': len(original_df) - len(processed_df),
            'removal_percentage': ((len(original_df) - len(processed_df)) / len(original_df) * 100) if len(original_df) > 0 else 0,
            'null_values_remaining': processed_df.isnull().sum().sum()
        }
        return report
