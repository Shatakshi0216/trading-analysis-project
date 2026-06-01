"""
Analysis module for trading analysis project.
Performs statistical analysis, correlations, and pattern discovery.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional


class TradingAnalyzer:
    """Perform trading data analysis."""
    
    @staticmethod
    def analyze_sentiment_trends(df: pd.DataFrame) -> Dict:
        """
        Analyze sentiment trends over time.
        
        Args:
            df: Sentiment DataFrame
            
        Returns:
            Dictionary with trend analysis
        """
        analysis = {}
        
        # Aggregate by month
        df['year_month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('year_month').agg({
            'fear_greed_value': ['mean', 'std', 'min', 'max'],
            'sentiment_category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        }).reset_index()
        
        analysis['monthly_trends'] = monthly.to_dict()
        
        # Volatility analysis
        analysis['volatility'] = {
            'overall': float(df['fear_greed_value'].std()),
            'mean': float(df['fear_greed_value'].mean()),
            'median': float(df['fear_greed_value'].median()),
            'range': float(df['fear_greed_value'].max() - df['fear_greed_value'].min())
        }
        
        return analysis
    
    @staticmethod
    def analyze_trader_performance(df: pd.DataFrame) -> Dict:
        """
        Analyze trader performance metrics.
        
        Args:
            df: Trader DataFrame
            
        Returns:
            Dictionary with performance metrics
        """
        analysis = {}
        
        # Overall metrics
        analysis['total_trades'] = int(len(df))
        analysis['total_pnl'] = float(df['pnl'].sum())
        analysis['average_pnl'] = float(df['pnl'].mean())
        analysis['std_pnl'] = float(df['pnl'].std())
        
        # Win rate
        profitable_trades = int((df['pnl'] > 0).sum())
        analysis['win_rate'] = float(profitable_trades / len(df) if len(df) > 0 else 0)
        
        # Profitable vs losing trades
        analysis['profitable_trades'] = profitable_trades
        analysis['losing_trades'] = int(len(df) - profitable_trades)
        
        # Risk metrics
        analysis['max_loss'] = float(df['pnl'].min())
        analysis['max_profit'] = float(df['pnl'].max())
        analysis['avg_loss'] = float(df[df['pnl'] < 0]['pnl'].mean() if (df['pnl'] < 0).any() else 0)
        analysis['avg_profit'] = float(df[df['pnl'] > 0]['pnl'].mean() if (df['pnl'] > 0).any() else 0)
        
        # Sharpe ratio
        if analysis['std_pnl'] != 0:
            analysis['sharpe_ratio'] = float((analysis['average_pnl'] / analysis['std_pnl']) * np.sqrt(252))
        else:
            analysis['sharpe_ratio'] = 0.0
        
        # Leverage analysis
        analysis['avg_leverage'] = float(df['leverage'].mean())
        analysis['max_leverage'] = float(df['leverage'].max())
        
        return analysis
    
    @staticmethod
    def analyze_sentiment_trader_correlation(merged_df: pd.DataFrame) -> Dict:
        """
        Analyze correlation between sentiment and trader performance.
        
        Args:
            merged_df: Merged sentiment and trader data
            
        Returns:
            Dictionary with correlation analysis
        """
        analysis = {}
        
        # Overall correlation
        if 'fear_greed_value' in merged_df.columns and 'pnl' in merged_df.columns:
            correlation = float(merged_df['fear_greed_value'].corr(merged_df['pnl']))
            analysis['overall_correlation'] = correlation
        
        # Correlation in different sentiment regimes
        fear_trades = merged_df[merged_df['fear_greed_value'] < 45]
        greed_trades = merged_df[merged_df['fear_greed_value'] > 55]
        neutral_trades = merged_df[(merged_df['fear_greed_value'] >= 45) & (merged_df['fear_greed_value'] <= 55)]
        
        analysis['fear_regime'] = {
            'trades': int(len(fear_trades)),
            'avg_pnl': float(fear_trades['pnl'].mean()),
            'win_rate': float((fear_trades['pnl'] > 0).sum() / len(fear_trades) if len(fear_trades) > 0 else 0),
            'volatility': float(fear_trades['pnl'].std())
        }
        
        analysis['greed_regime'] = {
            'trades': int(len(greed_trades)),
            'avg_pnl': float(greed_trades['pnl'].mean()),
            'win_rate': float((greed_trades['pnl'] > 0).sum() / len(greed_trades) if len(greed_trades) > 0 else 0),
            'volatility': float(greed_trades['pnl'].std())
        }
        
        analysis['neutral_regime'] = {
            'trades': int(len(neutral_trades)),
            'avg_pnl': float(neutral_trades['pnl'].mean()),
            'win_rate': float((neutral_trades['pnl'] > 0).sum() / len(neutral_trades) if len(neutral_trades) > 0 else 0),
            'volatility': float(neutral_trades['pnl'].std())
        }
        
        return analysis
    
    @staticmethod
    def identify_patterns(merged_df: pd.DataFrame) -> Dict:
        """
        Identify trading patterns in different market conditions.
        
        Args:
            merged_df: Merged sentiment and trader data
            
        Returns:
            Dictionary with identified patterns
        """
        patterns = {}
        
        # Time of day patterns
        if 'hour' in merged_df.columns:
            hourly_perf = merged_df.groupby('hour').agg({
                'pnl': ['mean', 'std', 'count'],
            }).round(3)
            patterns['hourly_patterns'] = hourly_perf.to_dict()
        
        # Day of week patterns
        if 'day_of_week' in merged_df.columns:
            daily_perf = merged_df.groupby('day_of_week').agg({
                'pnl': ['mean', 'std', 'count'],
            }).round(3)
            patterns['daily_patterns'] = daily_perf.to_dict()
        
        # Leverage impact
        if 'leverage' in merged_df.columns:
            high_leverage = merged_df[merged_df['leverage'] > 10]
            low_leverage = merged_df[merged_df['leverage'] <= 10]
            
            patterns['leverage_impact'] = {
                'high_leverage_avg_pnl': float(high_leverage['pnl'].mean()),
                'high_leverage_win_rate': float((high_leverage['pnl'] > 0).sum() / len(high_leverage) if len(high_leverage) > 0 else 0),
                'low_leverage_avg_pnl': float(low_leverage['pnl'].mean()),
                'low_leverage_win_rate': float((low_leverage['pnl'] > 0).sum() / len(low_leverage) if len(low_leverage) > 0 else 0)
            }
        
        return patterns
    
    @staticmethod
    def statistical_tests(merged_df: pd.DataFrame) -> Dict:
        """
        Perform statistical tests on trading data.
        
        Args:
            merged_df: Merged sentiment and trader data
            
        Returns:
            Dictionary with statistical test results
        """
        tests = {}
        
        # T-test: profitable vs losing trades
        if 'is_profitable' in merged_df.columns and 'fear_greed_value' in merged_df.columns:
            profitable = merged_df[merged_df['is_profitable'] == 1]['fear_greed_value']
            losing = merged_df[merged_df['is_profitable'] == 0]['fear_greed_value']
            
            if len(profitable) > 0 and len(losing) > 0:
                t_stat, p_value = stats.ttest_ind(profitable, losing)
                tests['sentiment_vs_profitability'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                }
        
        # Correlation significance test
        if 'fear_greed_value' in merged_df.columns and 'pnl' in merged_df.columns:
            correlation = merged_df['fear_greed_value'].corr(merged_df['pnl'])
            n = len(merged_df)
            if n > 2 and abs(correlation) < 1:
                t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation ** 2)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                tests['correlation_significance'] = {
                    'correlation': float(correlation),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                }
        
        return tests
