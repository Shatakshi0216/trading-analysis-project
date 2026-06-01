"""
Data loader module for trading analysis project.
Handles loading and initial validation of Bitcoin sentiment and Hyperliquid trader data.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Tuple, Dict, Optional


class DataLoader:
    """Load and validate trading datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = data_dir
        self.sentiment_data = None
        self.trader_data = None
    
    def load_sentiment_data(self, filename: str = "bitcoin_sentiment.csv") -> pd.DataFrame:
        """
        Load Bitcoin sentiment data.
        
        Args:
            filename: Name of sentiment CSV file
            
        Returns:
            DataFrame with sentiment data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Sentiment data not found at {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_cols = ['date', 'fear_greed_value', 'classification']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        self.sentiment_data = df
        print(f"✓ Loaded sentiment data: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def load_trader_data(self, filename: str = "hyperliquid_traders.csv") -> pd.DataFrame:
        """
        Load Hyperliquid trader data.
        
        Args:
            filename: Name of trader CSV file
            
        Returns:
            DataFrame with trader data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Trader data not found at {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_cols = ['timestamp', 'symbol', 'side', 'price', 'size', 'leverage', 'pnl']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        self.trader_data = df
        print(f"✓ Loaded trader data: {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all datasets.
        
        Returns:
            Tuple of (sentiment_data, trader_data)
        """
        sentiment = self.load_sentiment_data()
        trader = self.load_trader_data()
        return sentiment, trader
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of loaded data.
        
        Returns:
            Dictionary with data summaries
        """
        summary = {}
        
        if self.sentiment_data is not None:
            summary['sentiment'] = {
                'records': len(self.sentiment_data),
                'date_range': f"{self.sentiment_data['date'].min()} to {self.sentiment_data['date'].max()}",
                'fear_greed_stats': self.sentiment_data['fear_greed_value'].describe().to_dict(),
                'classifications': self.sentiment_data['classification'].value_counts().to_dict()
            }
        
        if self.trader_data is not None:
            summary['trader'] = {
                'records': len(self.trader_data),
                'time_range': f"{self.trader_data['timestamp'].min()} to {self.trader_data['timestamp'].max()}",
                'unique_symbols': self.trader_data['symbol'].nunique(),
                'pnl_stats': self.trader_data['pnl'].describe().to_dict(),
                'side_distribution': self.trader_data['side'].value_counts().to_dict()
            }
        
        return summary
